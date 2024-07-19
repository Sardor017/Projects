### Kirish
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer
import re


# Функция для загрузки и очистки данных
@st.cache_data
def load_and_clean_data(file_path):
    game = pd.read_csv(file_path, index_col=0)

    # Tozalash
    # "Name" ustunini tozalash
    game = game.dropna(subset=["Name"]).reset_index(drop=True)

    # "Publisher" ustunini tozalash
    Publisher_nan = game[game["Publisher"].isna()]
    Names = Publisher_nan["Name"].tolist()
    game_updated = game.copy()
    for name in Names:
        moda = game[game["Name"] == name]["Publisher"].mode()
        if not moda.empty:
            mode_value = moda[0]
            game_updated.loc[(game_updated["Name"] == name) & (game_updated["Publisher"].isna()), "Publisher"] = mode_value
    Publisher_nan = game[game["Publisher"].isna()]
    Genres = Publisher_nan["Genre"].unique()
    for genre in Genres:
        moda = game[game["Genre"] == genre]["Publisher"].mode()
        if not moda.empty:
            mode_value = moda[0]
            game_updated.loc[(game_updated["Genre"] == genre) & (game_updated["Publisher"].isna()), "Publisher"] = mode_value

    # "Year_of_Release" ustunini tozalash
    Year_nan = game[game["Year_of_Release"].isna()]
    Names = Year_nan["Name"].unique()
    for name in Names:
        moda = game[game["Name"] == name]["Year_of_Release"].mode()
        if not moda.empty:
            mode_value = moda[0]
            game_updated.loc[(game_updated["Name"] == name) & (game_updated["Year_of_Release"].isna()), "Year_of_Release"] = mode_value

    def extract_year_from_name(name):
        match = re.search(r'(\d{4})', name)
        if match:
            return int(match.group(1)) - 1
        return np.nan

    game_updated.loc[game_updated['Year_of_Release'].isna(), 'Year_of_Release'] = game_updated.loc[game_updated['Year_of_Release'].isna(), 'Name'].apply(extract_year_from_name)
    Year_nan = game_updated[game_updated["Year_of_Release"].isna()]
    Names = Year_nan["Name"].unique()
    for name in Names:
        median = game_updated[game_updated["Name"].str.contains(name)]["Year_of_Release"].median()
        if not np.isnan(median):
            game_updated.loc[(game_updated["Name"] == name) & (game_updated["Year_of_Release"].isna()), "Year_of_Release"] = median
    Year_nan = game_updated[game_updated["Year_of_Release"].isna()]
    Year_nan['Name'] = Year_nan['Name'].apply(lambda x: re.sub(r'\d+', '', x)).str.strip()
    Names = Year_nan["Name"].unique()
    for name in Names:
        median = game_updated[game_updated["Name"].str.contains(name)]["Year_of_Release"].median()
        if not np.isnan(median):
            game_updated.loc[(game_updated["Name"].str.contains(name)) & (game_updated["Year_of_Release"].isna()), "Year_of_Release"] = median
    Year_nan = game_updated[game_updated["Year_of_Release"].isna()]
    Year_nan["Name"] = Year_nan['Name'].str.split(':').str[0].str.strip()
    Names = Year_nan["Name"].unique()
    for name in Names:
        median = game_updated[game_updated["Name"].str.contains(name)]["Year_of_Release"].median()
        if not np.isnan(median):
            game_updated.loc[(game_updated["Name"].str.contains(name)) & (game_updated["Year_of_Release"].isna()), "Year_of_Release"] = median
    Year_nan = game_updated[game_updated["Year_of_Release"].isna()]
    Year_nan["Name"] = Year_nan['Name'].str.split(':').str[0].str.strip().str[0:-3].str.strip()
    Names = Year_nan["Name"].unique()
    for name in Names:
        try:
            median = game_updated[game_updated["Name"].str.contains(name)]["Year_of_Release"].median()
            if not np.isnan(median):
                game_updated.loc[(game_updated["Name"].str.contains(name)) & (game_updated["Year_of_Release"].isna()), "Year_of_Release"] = median
        except:
            pass
    game_updated["Year_of_Release"] = game_updated["Year_of_Release"].fillna(game_updated["Year_of_Release"].median())
    game_updated["Year_of_Release"] = game_updated["Year_of_Release"].round()
    game = game_updated.copy()

    # "Developer" ustunini to'ldirish
    game_updated['Developer'] = game_updated['Developer'].fillna(game_updated['Publisher'])

    # "Rating" ustunini to'ldirish
    Developers = game_updated["Developer"].unique()
    for devs in Developers:
        moda = game_updated[game_updated["Developer"] == devs]["Rating"].mode()
        if not moda.empty:
            mode_value = moda[0]
            game_updated.loc[(game_updated["Developer"] == devs) & (game_updated["Rating"].isna()), "Rating"] = mode_value
    columns = ["Genre", "Platform", "Year_of_Release", "Publisher"]
    for i in range(len(columns), 0, -1):
        mode_ratings = game_updated.groupby(columns[0:i])["Rating"].transform(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
        game_updated["Rating"] = game_updated["Rating"].fillna(mode_ratings)
    game_updated['Year_of_Release'] = game_updated['Year_of_Release'].astype(int)

    le_genre = LabelEncoder()
    game_updated['Genre_num'] = le_genre.fit_transform(game_updated['Genre'])
    le_rating = LabelEncoder()
    game_updated["Rating_num"] = le_rating.fit_transform(game_updated["Rating"])
    game_updated["User_Score"] = game_updated["User_Score"] * 10

    return game_updated

# Загрузка и очистка данных
file_path = "Video Games Salary.csv"
game_updated = load_and_clean_data(file_path)
game = pd.read_csv(file_path, index_col=0)
game
game_updated
# Streamlit UI
st.title("Video Game Sales Data Analysis")

# Chiqarish
region_names = {
    'NA_Sales': 'Amerikada',
    'EU_Sales': 'Yevropada',
    'JP_Sales': 'Yaponiyada',
    'Other_Sales': 'Boshqa hududlarda'
}
st.title("Video o'yinlarni Yil bo'yicha Sotish Hududlar kesimida")

years_sorted = sorted(game_updated['Year_of_Release'].unique())

# Выбор года выпуска
selected_year = st.selectbox("Ishlab chiqarilgan yilini tanlang", options=years_sorted)

# Фильтрация данных по выбранному году
selected_data = game_updated[game_updated['Year_of_Release'] == selected_year].iloc[0]

# Построение круговой диаграммы
fig, ax = plt.subplots(figsize=(8, 8))

# Подготовка данных для круговой диаграммы
sales_data = selected_data[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]
labels = [region_names[region] for region in sales_data.index]
sizes = sales_data.values
colors = ['#a2c2e1','#a4d4a1','#f2a6a6','#f7b27f']  # Цвета для круговой диаграммы

ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
ax.axis('equal')  # Сохраняет круговой формат

plt.title(f"{selected_year} yilda Video o'yinlarni Sotish")
st.pyplot(fig)





# Обработка пропущенных значений
columns_to_normalize = ['Critic_Score', 'Critic_Count', 'User_Score', 'User_Count']
game_updated[columns_to_normalize] = game_updated[columns_to_normalize].fillna(0)  # Замена NaN на 0

# 1. Стандартизация (Standardisation)
scaler = StandardScaler()
game_updated_standardized = game_updated.copy()
game_updated_standardized[columns_to_normalize] = scaler.fit_transform(game_updated[columns_to_normalize])

# 2. Средне-нормализация (Mean normalisation)
game_updated_mean_normalized = game_updated.copy()
for column in columns_to_normalize:
    game_updated_mean_normalized[column] = (game_updated[column] - game_updated[column].mean()) / (game_updated[column].max() - game_updated[column].min())

# 3. Масштабирование к максимуму и минимуму (Scaling to maximum and minimum)
min_max_scaler = MinMaxScaler()
game_updated_min_max_scaled = game_updated.copy()
game_updated_min_max_scaled[columns_to_normalize] = min_max_scaler.fit_transform(game_updated[columns_to_normalize])

# 4. Масштабирование к абсолютному максимуму (Scaling to absolute maximum)
max_abs_scaler = MaxAbsScaler()
game_updated_max_abs_scaled = game_updated.copy()
game_updated_max_abs_scaled[columns_to_normalize] = max_abs_scaler.fit_transform(game_updated[columns_to_normalize])

# 5. Масштабирование к медиане и квантили (Scaling to median and quantiles)
robust_scaler = RobustScaler()
game_updated_robust_scaled = game_updated.copy()
game_updated_robust_scaled[columns_to_normalize] = robust_scaler.fit_transform(game_updated[columns_to_normalize])

# 6. Масштабирование к единичной норме (Scaling to unit norm)
normalizer = Normalizer()
game_updated_normalized = game_updated.copy()
game_updated_normalized[columns_to_normalize] = normalizer.fit_transform(game_updated[columns_to_normalize])

# Streamlit UI
st.title("Video Game Sales Data Normalization and Standardization")

# Выбор метода нормализации
method = st.selectbox("Select Normalization/Standardization Method:", 
                      ["Standardisation", "Mean normalisation", "Scaling to maximum and minimum", 
                       "Scaling to absolute maximum", "Scaling to median and quantiles", "Scaling to unit norm"])

# Выбор данных для визуализации
if method == "Standardisation":
    data_to_plot = game_updated_standardized
elif method == "Mean normalisation":
    data_to_plot = game_updated_mean_normalized
elif method == "Scaling to maximum and minimum":
    data_to_plot = game_updated_min_max_scaled
elif method == "Scaling to absolute maximum":
    data_to_plot = game_updated_max_abs_scaled
elif method == "Scaling to median and quantiles":
    data_to_plot = game_updated_robust_scaled
else:
    data_to_plot = game_updated_normalized

# Построение графиков
fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# Построение точечного графика для Critic_Score
sns.scatterplot(data=data_to_plot, x='Critic_Score', y='User_Score', hue='Year_of_Release', ax=axs[0], palette='viridis', s=100)
axs[0].set_title("Critic Score vs User Score")
axs[0].set_xlabel("Critic Score")
axs[0].set_ylabel("User Score")
axs[0].legend(title='Year of Release')

# Построение гистограммы для User_Score
sns.histplot(data=data_to_plot['User_Score'].dropna(), bins=10, kde=True, ax=axs[1], color='green')
axs[1].set_title("Distribution of User Score")
axs[1].set_xlabel("User Score")
axs[1].set_ylabel("Frequency")

st.pyplot(fig)
























melted_data = pd.melt(game_updated, id_vars=['Genre', 'Year_of_Release'], 
                      value_vars=['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'],
                      var_name='Region', value_name='Sales')

# Streamlit UI
st.title("Video o'yinlarni yillar, janr hamda sotish joyi kesimida Sotilishi")

# Выбор года выпуска
years = st.multiselect("Ishlab chiqarilgan yilini tanlang", 
                       options=melted_data['Year_of_Release'].unique(), 
                       default=melted_data['Year_of_Release'].unique(), 
                       key='years_multiselect')

# Выбор жанра
genres = st.multiselect("Janrni tanlang", 
                       options=melted_data['Genre'].unique(), 
                       default=melted_data['Genre'].unique(), 
                       key='genres_multiselect')

# Выбор региона
regions = st.multiselect("Qayerda sotilganini tanlang", 
                         options=['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'], 
                         default=['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'], 
                         key='regions_multiselect')

# Фильтрация данных
filtered_data = melted_data[
    melted_data['Year_of_Release'].isin(years) &
    melted_data['Genre'].isin(genres) &
    melted_data['Region'].isin(regions)
]

# Выбор типа графика
plot_type = st.selectbox("Grafik turini tanlang", ["Bar Plot", "Histogram", "Box Plot", "Line Plot", "Heatmap"], key='plot_type_selectbox')

# Построение графика
fig, ax = plt.subplots(figsize=(12, 7))

if plot_type == "Bar Plot":
    sns.barplot(data=filtered_data, x='Genre', y='Sales', hue='Region', ax=ax)
    ax.set_title("Video o'yinlarni yillar, janr hamda sotish joyi kesimida Sotilishi")

elif plot_type == "Histogram":
    sns.histplot(data=filtered_data, x='Sales', hue='Region', multiple='stack', ax=ax)
    ax.set_title("Video o'yinlarni yillar, janr hamda sotish joyi kesimida Sotilishi")

elif plot_type == "Box Plot":
    sns.boxplot(data=filtered_data, x='Genre', y='Sales', hue='Region', ax=ax)
    ax.set_title("Video o'yinlarni yillar, janr hamda sotish joyi kesimida Sotilishi")

elif plot_type == "Line Plot":
    sns.lineplot(data=filtered_data, x='Year_of_Release', y='Sales', hue='Region', ax=ax, marker='o')
    ax.set_title("Video o'yinlarni yillar, janr hamda sotish joyi kesimida Sotilishi")

elif plot_type == "Heatmap":
    pivot_data = filtered_data.pivot_table(index='Genre', columns='Region', values='Sales', aggfunc='sum')
    sns.heatmap(pivot_data, cmap='YlGnBu', annot=True, fmt='.1f', ax=ax)
    ax.set_title("Video o'yinlarni yillar, janr hamda sotish joyi kesimida Sotilishi")

plt.xticks(rotation=45)
st.pyplot(fig)









st.title("Video o'yinlarni O'zgaruvchilar Taqqoslash")

# Построение гистограммы для оценок критиков и пользователей
fig, ax = plt.subplots(figsize=(12, 6))

# Построение гистограммы для оценок критиков
sns.histplot(game_updated['Critic_Score'].dropna(), bins=10, color='blue', label='Critic Score', ax=ax)

# Построение гистограммы для оценок пользователей
sns.histplot(game_updated['User_Score'].dropna(), bins=10, color='green', label='User Score', ax=ax)

ax.set_title("O'zgaruvchilar Taqqoslash: Critic Score vs User Score")
ax.set_xlabel("O'zgaruvchilar")
ax.set_ylabel("Son")
ax.legend()

st.pyplot(fig)






st.title("Video o'yinlarni narxiga qarab berilgan baholar Sotish joyiga qarab")
region_names = {
    'NA_Sales': 'Amerikada',
    'EU_Sales': 'Yevropada',
    'JP_Sales': 'Yaponiyada',
    'Other_Sales': 'Boshqa hududlarda'
}
reverse_region_names = {v: k for k, v in region_names.items()}
# Выбор региона
region_selection = st.selectbox("Tanlangan mintaqa:", options=list(region_names.values()))
region = reverse_region_names[region_selection]
# Построение графиков
fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# Построение точечного графика для выбранного региона и оценок критиков
sns.scatterplot(data=game_updated, x=region, y='Critic_Score', hue='Year_of_Release', ax=axs[0], palette='viridis', s=100)
axs[0].set_title(f"{region_names[region]} vs Mutahassislarlar bahosi")
axs[0].set_xlabel(f"{region_names[region]} (Million Dollarda)")
axs[0].set_ylabel("Mutahassislarlar bahosi")
axs[0].legend(title='Ishlab chiqilgan yili')

# Построение точечного графика для выбранного региона и оценок пользователей
sns.scatterplot(data=game_updated, x=region, y='User_Score', hue='Year_of_Release', ax=axs[1], palette='viridis', s=100)
axs[1].set_title(f"{region_names[region]} vs Foydalanuvchilar bahosi")
axs[1].set_xlabel(f"{region_names[region]} (Million Dollarda)")
axs[1].set_ylabel("Foydalanuvchilar bahosi")
axs[1].legend(title='Ishlab chiqilgan yili')

st.pyplot(fig)