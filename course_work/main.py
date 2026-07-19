import pandas as pd
import numpy as np
import os
import warnings
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.preprocessing import LabelEncoder
from implicit.als import AlternatingLeastSquares
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP

# Новые библиотеки для доработок
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import lightgbm as lgb
from collections import defaultdict

warnings.filterwarnings('ignore')

import sys
import logging
from datetime import datetime
import builtins


# НАСТРОЙКА ЛОГИРОВАНИЯ

# Создаем имя файла лога с датой и временем
log_filename = f"recommendation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

# Настраиваем логгер с выводом в файл и в консоль
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Перехватываем функцию print для автоматического логирования
original_print = print
def print_with_logging(*args, **kwargs):
    """Обертка над print, которая автоматически логирует весь вывод"""
    message = ' '.join(map(str, args))
    logger.info(message)
    original_print(*args, **kwargs)

# Заменяем встроенную функцию print
builtins.print = print_with_logging

print(f"✅ Логирование включено. Файл лога: {log_filename}")

# МЕТРИКИ

def ndcg_at_k(recommended_items, test_items, k=10):
    """Normalized Discounted Cumulative Gain. Учитывает порядок рекомендаций."""
    recommended_items = recommended_items[:k]
    if not test_items:
        return 0.0

    dcg = 0.0
    for i, item in enumerate(recommended_items):
        if item in test_items:
            dcg += 1.0 / np.log2(i + 2)

    idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(test_items), k))])
    return dcg / idcg if idcg > 0 else 0.0


def catalog_coverage(all_recommended_items, total_items_in_catalog):
    """Какую долю каталога модель вообще когда-либо рекомендует."""
    unique_recommended = len(set(all_recommended_items))
    return unique_recommended / total_items_in_catalog


def category_diversity(recommended_items, item_to_category_dict, k=10):
    """Разнообразие: доля уникальных категорий в топ-K рекомендациях."""
    recommended_items = recommended_items[:k]
    categories = [item_to_category_dict.get(item, -1) for item in recommended_items]
    unique_cats = len(set([c for c in categories if c != -1]))
    return unique_cats / k if k > 0 else 0.0


def evaluate_model_advanced(model, user_items_train, user_items_test, item_to_category_dict, k=10):
    """Расширенная оценка модели: Recall, NDCG, Coverage, Diversity."""
    n_users = user_items_test.shape[0]
    total_recall = 0.0
    total_ndcg = 0.0
    total_diversity = 0.0
    n_evaluated_users = 0
    all_recommended_items = []

    for user_id in tqdm(range(n_users), desc="Оценка метрик", disable=False):
        if user_items_test[user_id].nnz == 0:
            continue

        recommended_items, _ = model.recommend(
            user_id,
            user_items_train[user_id],
            N=k,
            filter_already_liked_items=True
        )

        test_items = set(user_items_test[user_id].nonzero()[1])
        if not test_items:
            continue

        # Recall
        hits = len(set(recommended_items) & test_items)
        recall = hits / len(test_items)
        total_recall += recall

        # NDCG
        ndcg = ndcg_at_k(recommended_items, test_items, k)
        total_ndcg += ndcg

        # Diversity
        div = category_diversity(recommended_items, item_to_category_dict, k)
        total_diversity += div

        all_recommended_items.extend(recommended_items)
        n_evaluated_users += 1

    # Coverage
    coverage = catalog_coverage(all_recommended_items, user_items_test.shape[1])

    metrics = {
        'Recall@K': total_recall / n_evaluated_users if n_evaluated_users > 0 else 0.0,
        'NDCG@K': total_ndcg / n_evaluated_users if n_evaluated_users > 0 else 0.0,
        'Coverage': coverage,
        'Diversity': total_diversity / n_evaluated_users if n_evaluated_users > 0 else 0.0
    }
    return metrics

# FEATURE ENGINEERING

def apply_time_decay_and_sessions(events_df):
    """
    1. Применяет временное затухание (Time Decay).
    2. Разбивает историю на сессии (пауза > 30 минут = новая сессия).
    """
    df = events_df.copy()
    df = df.sort_values(['visitorid', 'timestamp'])

    # 1. Time Decay (Затухание)
    max_ts = df['timestamp'].max()
    df['days_ago'] = (max_ts - df['timestamp']) / (1000 * 3600 * 24)
    decay_lambda = 0.1
    df['decay_weight'] = np.exp(-decay_lambda * df['days_ago'])

    # Итоговый вес = базовый вес * затухание
    base_weights = {'view': 1, 'addtocart': 3, 'transaction': 5}
    df['base_weight'] = df['event'].map(base_weights)
    df['final_weight'] = df['base_weight'] * df['decay_weight']

    # 2. Разметка сессий
    df['time_diff'] = df.groupby('visitorid')['timestamp'].diff()
    session_gap_ms = 30 * 60 * 1000
    df['is_new_session'] = (df['time_diff'] > session_gap_ms) | (df['time_diff'].isna())
    df['session_id'] = df.groupby('visitorid')['is_new_session'].cumsum()

    print(f"Средний вес после затухания: {df['final_weight'].mean():.3f}")
    print(f"Выделено сессий: {df['session_id'].nunique()}")

    return df

# ДВУХСТУПЕНЧАТАЯ АРХИТЕКТУРА

class TwoStageRecSys:
    """ALS (Retrieval) + LightGBM (Ranking)"""

    def __init__(self, n_factors=32):
        self.als_model = AlternatingLeastSquares(
            factors=n_factors,
            regularization=0.01,
            iterations=15,
            random_state=42,
            use_gpu=False
        )
        self.ranker = None

    def fit_retrieval(self, train_matrix):
        """Stage 1: Обучение ALS"""
        print("Обучение ALS (Retrieval)...")
        self.als_model.fit(train_matrix)

    def generate_training_data(self, events_df, n_negatives=5):
        """Генерирует датасет для обучения ранжировщика"""
        print("Генерация признаков для LightGBM...")

        # Позитивные примеры (покупки/добавления)
        positives = events_df[events_df['event'].isin(['addtocart', 'transaction'])].copy()

        # Статистика для признаков
        item_popularity = events_df['item_id'].value_counts().to_dict()
        user_activity = events_df['user_id'].value_counts().to_dict()

        rows = []

        # Позитивные примеры
        for _, row in positives.iterrows():
            als_score = self.als_model.recommend(
                row['user_id'],
                csr_matrix((1, self.als_model.item_factors.shape[0])),
                N=1,
                filter_already_liked_items=False
            )[1][0] if hasattr(self.als_model, 'item_factors') else 0.0

            rows.append({
                'user_id': row['user_id'],
                'item_id': row['item_id'],
                'als_score': als_score,
                'item_popularity': item_popularity.get(row['item_id'], 0),
                'user_activity': user_activity.get(row['user_id'], 0),
                'target': 1
            })

        # Негативные примеры (Random Negative Sampling)
        all_items = set(events_df['item_id'].unique())
        np.random.seed(42)

        for user_id in positives['user_id'].unique():
            user_items = set(positives[positives['user_id'] == user_id]['item_id'])
            available_negatives = list(all_items - user_items)

            if len(available_negatives) >= n_negatives:
                sampled_negatives = np.random.choice(available_negatives, n_negatives, replace=False)

                for neg_item in sampled_negatives:
                    rows.append({
                        'user_id': user_id,
                        'item_id': neg_item,
                        'als_score': 0.0,
                        'item_popularity': item_popularity.get(neg_item, 0),
                        'user_activity': user_activity.get(user_id, 0),
                        'target': 0
                    })

        return pd.DataFrame(rows)

    def fit_ranker(self, features_df):
        """Stage 2: Обучение LightGBM"""
        print("Обучение LightGBM (Ranking)...")
        X = features_df.drop('target', axis=1)
        y = features_df['target']

        self.ranker = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            verbose=-1
        )
        self.ranker.fit(X, y)

    def predict(self, user_id, candidate_items, user_items_train):
        """Финальное ранжирование кандидатов"""
        if self.ranker is None:
            raise ValueError("Ranker не обучен")

        item_popularity = {i: 0 for i in candidate_items}  # Заглушка
        user_activity = 0  # Заглушка

        features = pd.DataFrame({
            'user_id': [user_id] * len(candidate_items),
            'item_id': candidate_items,
            'als_score': [0.0] * len(candidate_items),  # Заглушка
            'item_popularity': [item_popularity.get(i, 0) for i in candidate_items],
            'user_activity': [user_activity] * len(candidate_items)
        })

        scores = self.ranker.predict_proba(features)[:, 1]
        sorted_indices = np.argsort(scores)[::-1]

        return [candidate_items[i] for i in sorted_indices]

# SEQUENTIAL RECOMMENDATIONS - SASRec

class SessionDataset(Dataset):
    """Датасет для SASRec: последовательности сессий"""

    def __init__(self, sessions, max_len=50):
        self.sessions = sessions
        self.max_len = max_len

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        session = self.sessions[idx]

        # Padding или обрезка
        if len(session) > self.max_len:
            session = session[-self.max_len:]
        else:
            session = [0] * (self.max_len - len(session)) + session

        return torch.tensor(session, dtype=torch.long)


class SASRec(nn.Module):
    """Self-Attentive Sequential Recommendation"""

    def __init__(self, num_items, hidden_dim=64, num_heads=2, num_layers=2, dropout=0.2, max_len=50):
        super(SASRec, self).__init__()
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.max_len = max_len

        self.item_emb = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, log_seqs):
        seq_len = log_seqs.size(1)
        positions = torch.arange(seq_len, dtype=torch.long, device=log_seqs.device).unsqueeze(0)

        x = self.item_emb(log_seqs) + self.pos_emb(positions)
        x = self.dropout(x)

        attn_mask = self._get_causal_mask(seq_len).to(log_seqs.device)
        x = self.transformer(x, mask=attn_mask)
        x = self.layer_norm(x)

        return x

    def _get_causal_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask

    def predict(self, log_seqs):
        hidden_states = self.forward(log_seqs)
        last_hidden = hidden_states[:, -1, :]
        logits = torch.matmul(last_hidden, self.item_emb.weight.T)
        return logits


def train_sasrec(model, dataloader, epochs=10, lr=0.001, device='cpu'):
    """Обучение SASRec"""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            batch = batch.to(device)

            # Вход: все кроме последнего, Таргет: последний
            inputs = batch[:, :-1]
            targets = batch[:, -1]

            optimizer.zero_grad()
            logits = model.predict(inputs)
            loss = criterion(logits, targets)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")


def evaluate_sasrec(model, test_sessions, item_to_category_dict, k=10, device='cpu'):
    """Оценка SASRec"""
    model.eval()
    total_recall = 0.0
    total_ndcg = 0.0
    total_diversity = 0.0
    n_evaluated = 0

    with torch.no_grad():
        for session in tqdm(test_sessions, desc="Оценка SASRec"):
            if len(session) < 2:
                continue

            if len(session) > model.max_len + 1:
                session = session[-(model.max_len + 1):]

            inputs = torch.tensor([session[:-1]], dtype=torch.long).to(device)
            target = session[-1]

            logits = model.predict(inputs)
            logits[0, 0] = float('-inf')  # Исключаем padding

            _, recommended = torch.topk(logits, k, dim=1)
            recommended = recommended[0].cpu().numpy()

            # Recall
            if target in recommended:
                total_recall += 1.0

            # NDCG
            ndcg = ndcg_at_k(recommended, [target], k)
            total_ndcg += ndcg

            # Diversity
            div = category_diversity(recommended, item_to_category_dict, k)
            total_diversity += div

            n_evaluated += 1

    metrics = {
        'Recall@K': total_recall / n_evaluated if n_evaluated > 0 else 0.0,
        'NDCG@K': total_ndcg / n_evaluated if n_evaluated > 0 else 0.0,
        'Diversity': total_diversity / n_evaluated if n_evaluated > 0 else 0.0
    }
    return metrics

# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ

def load_item_categories():
    files = ['./resources/item_properties_part1.csv', './resources/item_properties_part2.csv']
    dfs = []
    for f in files:
        if os.path.exists(f):
            df = pd.read_csv(f)
            dfs.append(df)
        else:
            print(f"Файл {f} не найден")
            return None

    props = pd.concat(dfs, ignore_index=True)
    cat_props = props[props['property'] == 'categoryid'].copy()
    cat_props['categoryid'] = pd.to_numeric(cat_props['value'], errors='coerce')
    cat_props = cat_props.dropna(subset=['categoryid'])
    cat_props['categoryid'] = cat_props['categoryid'].astype(int)
    item_to_category = cat_props[['itemid', 'categoryid']].drop_duplicates()
    print(f"Загружено {len(item_to_category)} товаров с категориями")
    return item_to_category


def visualize_item_embeddings(model, item_encoder, item_to_category, n_samples=3000):
    print("Визуализация эмбеддингов товаров")
    item_indices = np.arange(len(model.item_factors))
    original_itemids = item_encoder.inverse_transform(item_indices)
    df_items = pd.DataFrame({'itemid': original_itemids, 'internal_id': item_indices})
    df_items = df_items.merge(item_to_category, on='itemid', how='left')
    df_items = df_items.dropna(subset=['categoryid'])
    df_items['categoryid'] = df_items['categoryid'].astype(int)

    top_cats = df_items['categoryid'].value_counts().head(10).index
    df_items = df_items[df_items['categoryid'].isin(top_cats)]

    if len(df_items) > n_samples:
        df_items = df_items.sample(n=n_samples, random_state=42)

    if len(df_items) == 0:
        print("Нет данных для визуализации")
        return

    item_factors = model.item_factors[df_items['internal_id'].values]
    print("Запуск UMAP")
    embeddings_2d = UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1).fit_transform(item_factors)
    df_items['x'], df_items['y'] = embeddings_2d[:, 0], embeddings_2d[:, 1]

    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df_items, x='x', y='y', hue='categoryid', palette='tab10', alpha=0.7, s=25)
    plt.title("Эмбеддинги товаров (Baseline ALS)", fontsize=14)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend(title="Категория", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("item_embeddings_umap.png", dpi=150)
    plt.show()
    print("График сохранён")

# ОСНОВНОЙ БЛОК

if __name__ == "__main__":
    print("=" * 80)
    print("ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ")
    print("=" * 80)

    events = pd.read_csv('./resources/events.csv')
    events = events[events['event'].isin(['view', 'addtocart', 'transaction'])].copy()
    print(f"Загружено {len(events)} событий")

    # Фильтрация пользователей
    user_counts = events['visitorid'].value_counts()
    valid_visitors = user_counts[user_counts >= 2].index
    events = events[events['visitorid'].isin(valid_visitors)].copy()
    print(f"После фильтрации: {len(events)} событий")

    # Кодирование
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    events['user_id'] = user_encoder.fit_transform(events['visitorid'])
    events['item_id'] = item_encoder.fit_transform(events['itemid'])
    n_users = events['user_id'].nunique()
    n_items = events['item_id'].nunique()
    print(f"Кодировано: {n_users} пользователей, {n_items} товаров")

    # ПРИМЕНЕНИЕ TIME DECAY И СЕССИЙ

    print("\n" + "=" * 80)
    print("ПРИМЕНЕНИЕ TIME DECAY И СЕССИЙ")
    print("=" * 80)
    events = apply_time_decay_and_sessions(events)

    # РАЗДЕЛЕНИЕ НА TRAIN/TEST

    print("\n" + "=" * 80)
    print("РАЗДЕЛЕНИЕ НА TRAIN/TEST")
    print("=" * 80)
    events = events.sort_values('timestamp')
    last_event_idx = events.groupby('user_id')['timestamp'].idxmax()
    test_events = events.loc[last_event_idx]
    train_events = events.drop(index=last_event_idx)
    print(f"Обучающих: {len(train_events)} событий, Тестовых: {len(test_events)} событий.")


    # Построение матриц
    def build_matrix(df, n_u, n_i, weight_col='final_weight'):
        return coo_matrix(
            (df[weight_col].values, (df['user_id'].values, df['item_id'].values)),
            shape=(n_u, n_i)
        ).tocsr()


    train_matrix = build_matrix(train_events, n_users, n_items)
    test_matrix = build_matrix(test_events, n_users, n_items)

    # Загрузка категорий для метрик
    item_to_category = load_item_categories()
    if item_to_category is not None:
        item_to_category_internal = item_to_category.copy()
        known_items = set(item_encoder.classes_)
        items_before = len(item_to_category_internal)
        item_to_category_internal = item_to_category_internal[
            item_to_category_internal['itemid'].isin(known_items)
        ]
        items_after = len(item_to_category_internal)
        print(f"Категории: оставлено {items_after} из {items_before} товаров "
              f"(отфильтровано {items_before - items_after} неизвестных)")

        item_to_category_internal['item_id'] = item_encoder.transform(item_to_category_internal['itemid'])
        item_to_cat_dict = dict(zip(item_to_category_internal['item_id'], item_to_category_internal['categoryid']))
    else:
        item_to_cat_dict = {}

    # BASELINE: ALS

    print("\n" + "=" * 80)
    print("BASELINE: ALS")
    print("=" * 80)

    model = AlternatingLeastSquares(
        factors=32,
        regularization=0.01,
        iterations=15,
        random_state=42,
        use_gpu=False
    )
    model.fit(train_matrix)
    print("Baseline ALS обучена")

    # Оценка Baseline
    train_binary = train_matrix.copy()
    train_binary.data = np.ones_like(train_binary.data)
    test_binary = test_matrix.copy()
    test_binary.data = np.ones_like(test_binary.data)

    baseline_metrics = evaluate_model_advanced(model, train_binary, test_binary, item_to_cat_dict, k=10)
    print("\nBaseline ALS метрики:")
    for metric, value in baseline_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Визуализация
    if item_to_category is not None:
        visualize_item_embeddings(model, item_encoder, item_to_category)

    # TWO-STAGE: ALS + LightGBM

    print("\n" + "=" * 80)
    print("TWO-STAGE: ALS + LightGBM")
    print("=" * 80)

    two_stage = TwoStageRecSys(n_factors=32)
    two_stage.fit_retrieval(train_matrix)

    # Генерация данных для ранжировщика
    features_df = two_stage.generate_training_data(train_events, n_negatives=5)
    print(f"Датасет для LightGBM: {len(features_df)} примеров")

    # Обучение ранжировщика
    two_stage.fit_ranker(features_df)
    print("Two-Stage модель обучена")

    # Оценка Two-Stage (упрощенная)
    print("\nTwo-Stage метрики:")
    print("  (Для полной оценки нужно реализовать recommend() метод)")
    print("  LightGBM улучшает ранжирование кандидатов от ALS")


    # SEQUENTIAL: SASRec

    print("\n" + "=" * 80)
    print("SEQUENTIAL: SASRec")
    print("=" * 80)

    # Подготовка сессий для SASRec
    print("Подготовка сессий для SASRec...")
    sessions = []
    for user_id, group in train_events.groupby('user_id'):
        user_sessions = group.groupby('session_id')['item_id'].apply(list).tolist()
        sessions.extend([s for s in user_sessions if len(s) >= 2])

    print(f"Всего сессий: {len(sessions)}")

    # Разделение на train/test для SASRec
    np.random.seed(42)
    np.random.shuffle(sessions)
    split_idx = int(len(sessions) * 0.8)
    train_sessions = sessions[:split_idx]
    test_sessions = sessions[split_idx:]

    print(f"Train сессий: {len(train_sessions)}, Test сессий: {len(test_sessions)}")

    # Создание DataLoader
    train_dataset = SessionDataset(train_sessions, max_len=50)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    # Обучение SASRec
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Используется устройство: {device}")

    sasrec_model = SASRec(
        num_items=n_items,
        hidden_dim=64,
        num_heads=2,
        num_layers=2,
        dropout=0.2,
        max_len=50
    )

    train_sasrec(sasrec_model, train_loader, epochs=5, lr=0.001, device=device)

    # Оценка SASRec
    sasrec_metrics = evaluate_sasrec(sasrec_model, test_sessions, item_to_cat_dict, k=10, device=device)
    print("\nSASRec метрики:")
    for metric, value in sasrec_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # СРАВНЕНИЕ РЕЗУЛЬТАТОВ

    print("\n" + "=" * 80)
    print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
    print("=" * 80)

    comparison = pd.DataFrame({
        'Метрика': list(baseline_metrics.keys()),
        'Baseline ALS': list(baseline_metrics.values()),
        'SASRec': [sasrec_metrics.get(k, 0) for k in baseline_metrics.keys()]
    })

    print(comparison.to_string(index=False))

    # Визуализация сравнения
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics_to_plot = ['Recall@K', 'NDCG@K', 'Diversity']

    for i, metric in enumerate(metrics_to_plot):
        baseline_val = baseline_metrics[metric]
        sasrec_val = sasrec_metrics.get(metric, 0)

        axes[i].bar(['Baseline ALS', 'SASRec'], [baseline_val, sasrec_val], color=['steelblue', 'coral'])
        axes[i].set_ylabel(metric)
        axes[i].set_title(f'{metric} Сравнение')
        axes[i].set_ylim(0, max(baseline_val, sasrec_val) * 1.2)

        for j, v in enumerate([baseline_val, sasrec_val]):
            axes[i].text(j, v + 0.001, f'{v:.4f}', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150)
    plt.show()
    print("График сравнения сохранён")
