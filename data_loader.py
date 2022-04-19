import pandas as pd
import pandas.api.types as ptypes
import numpy as np
from scipy.interpolate import interp1d

from sqlalchemy import create_engine  # database connection
from IPython.display import display
import plotly.express as px
import matplotlib.pyplot as plt
import subprocess
from sklearn.neighbors import KDTree
monthDict = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct',
             11: 'Nov', 12: 'Dec'}

seasonDict = {1: 'Winter  ',
              2: 'Spring',
              3: 'Summer ',
              4: 'Fall'
              }


def create_animation(bird, remove_noise=True, by_season=True):
    center = dict(lat=bird.lat.median(), lon=bird.lon.median())
    bird['season'] = bird.index.month % 12 // 3 + 1

    for year in [1970, 1980, 1990, 2000, 2010]:
        decade = bird[((bird.index.year >= year) & (bird.index.year < year + 10))]
        label = str(year)[2:] + 's '
        decade.insert(2, 'decade', label, True)
        dfs = []
        if by_season:
            if remove_noise:
                coords = decade[['lat', 'lon']]
                tree = KDTree(coords, leaf_size=3)
                decade['kd'] = tree.query_radius(coords, r=0.3, count_only=True)
                decade = decade[decade.kd > 200]

            for i, df in decade.groupby(by=[decade.season]):
                df.decade = df.decade + seasonDict[i]
                df.season = seasonDict[i]
                dfs.append(df)
                render_densmap(pd.concat(dfs), center, year, i, by='season')
        else:
            for month in range(1, 13):
                df = decade[decade.index.month == month]
                df.decade = df.decade + monthDict[month]
                df['month'] = monthDict[month]
                dfs.append(df)

                render_densmap(pd.concat(dfs).sort_values(by='month'), center, year, month)
        display('{} completed'.format(label))
    step = 'season' if by_season else 'month'
    noise = 'denoised' if by_season else None
    subprocess.run(["convert", "-delay", "100", "anim/*{}.png".format(step), "anim/out_{}_{}.gif".format(step, noise)])


def render_densmap(bird, center, year, month, by='month'):
    if by == 'season':
        fig = px.scatter_mapbox(
            bird,
            lat='lat', lon='lon',
            center=center,
            zoom=2,
            color='season',
            mapbox_style="stamen-terrain",
            hover_name='decade'
        )
    else:
        fig = px.scatter_mapbox(
            bird,
            lat='lat', lon='lon',
            center=center,
            zoom=2,
            color='month',
            mapbox_style="stamen-terrain",
            hover_name='decade'
        )
    fig.write_image("anim/fig_{}_{}_{}.png".format(year, month, by))


def plot_densmap(bird):
    bird.sort_index(inplace=True)
    center = dict(lat=bird.lat.median(), lon=bird.lon.median())
    fig = px.density_mapbox(bird, lat='lat', lon='lon', radius=4,
                            center=center, zoom=2,
                            mapbox_style="stamen-terrain")
    fig.show()

    # fig = px.density_mapbox(bird, lat='lat', lon='lon', radius=4,
    #                         center=center, zoom=2,
    #                         mapbox_style="stamen-terrain")


def plot_sightings(bird, frame='Yearly'):  # frame OPTIONS: {Daily, Monthly OR Yearly}
    assert ptypes.is_datetime64_any_dtype(bird.index)
    if frame == 'Yearly':
        df = bird.groupby(by=[bird.index.year]).count()[['event']]
        df = df.rename({'event': 'count'}, axis=1)
        display(df.head())
        ax = df['count'].plot(kind='area', rot=25, use_index=True, title='Yearly Sightings')

        mu = df['count'].mean()
        med = df['count'].median()

        plt.text(1965, mu + 100, f'average: {int(mu)}', va='center')
        ax.axhline(mu, color='black', alpha=.4, animated=True)

        plt.text(1965, med + 100, f'median {int(med)}', va='center', color='red')
        ax.axhline(med, color='red', alpha=.4)

        # new_x = np.arange(df.index.min(), df.index.max(), 2)
        # cubic_interp = interp1d(df.index.values, df['count'].values, kind='cubic')
        # cubic_results = cubic_interp(new_x)
        # ax.plot(new_x, cubic_results, 'y')
        plt.show()

    elif frame == 'Monthly':
        df = bird.groupby(by=[bird.index.month_name()]).count()[['event']]
        df = df.rename({'event': 'count'}, axis=1)
        months = ["January", "February", "March", "April", "May", "June",
                  "July", "August", "September", "October", "November", "December"]
        df.index = pd.Categorical(df.index, categories=months, ordered=True)
        df.sort_index(inplace=True)
        display(df.head())
        df['count'].plot(kind='bar', use_index=True, rot=25, title='Monthly Sightings')

    elif frame == 'Season':
        bird['season'] = bird.index.month % 12 // 3 + 1
        df = bird.groupby(by=[bird.season]).count()[['event']]
        df = df.rename(index={1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'})
        df = df.rename({'event': 'count'}, axis=1)
        display(df.head())
        df['count'].plot(kind='bar', use_index=True, rot=25, title='Sightings')

    elif frame == 'Daily':
        df = bird.groupby(by=[bird.index.day]).count()[['event']]
        df = df.rename({'event': 'count'}, axis=1)
        display(df.head())
        df['count'].plot(kind='area', use_index=True, rot=25, title='Daily Sightings')
    elif frame == 'WeekDay':
        df = bird.groupby(by=[bird.index.day_name()]).count()[['event']]
        df = df.rename({'event': 'count'}, axis=1)
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        df.index = pd.Categorical(df.index, categories=days, ordered=True)
        df.sort_index(inplace=True)
        display(df.head())
        df['count'].plot(kind='bar', use_index=True, rot=25, title='Sightings by Day of Week')

    else:
        print('ALLOWED OPTIONS: {Daily, WeekDay, Monthly OR Yearly}')


class Data:
    def __init__(self, bird_name=None, bird_id=None, popular=False, random=False, group_n=3):
        self.disk_engine = create_engine(
            'sqlite:///data/db/bbp_{}.db'.format(group_n))  # Initializes database in current directory

        self.bird_list = pd.read_csv('data/readme_NABBP_bird_groups_1-10_and_MALL.csv',
                                     usecols=['Download_grp', 'SPECIES_ID', 'SPECIES_NAME'],
                                     index_col='SPECIES_ID')
        self.bird_id = bird_id
        self.birds = self.get_popular()
        if bird_name:
            self.bird_id = int(self.find_bird_ids(bird_name).index[0])
        elif popular:
            self.bird_id = self.birds.index.iloc[0]
        elif random:
            self.bird_id = self.get_random()
        if isinstance(self.bird_id, int):
            self.load_bird()
        else:
            self.bird_name = None
            self.bird = None

    def find_bird_ids(self, bird_name: str = 'Eagle'):  # all birds with keyword eagle
        return self.bird_list[self.bird_list.SPECIES_NAME.str.contains(bird_name)]

    def find_bird_name(self, bird_id: int = 3850):
        display(self.birds[self.birds.index == bird_id])
        return self.birds[self.birds.index == bird_id].SPECIES_NAME.iloc[0]

    def load_bird(self):
        if self.bird_id:
            self.bird_name = self.find_bird_name(self.bird_id)
            df = pd.read_sql_query(f'SELECT * FROM data WHERE id={self.bird_id}', self.disk_engine, parse_dates=['t'],
                                   index_col='t')
            self.bird = df.drop(['index'], axis=1).drop_duplicates()

    def get_popular(self):
        occurs = pd.read_sql_query(f'SELECT id as SPECIES_ID, count(*) as cnt FROM data group by id', self.disk_engine)
        occurs.set_index('SPECIES_ID', inplace=True)
        df = self.bird_list.join(occurs.cnt)
        df = df[df.cnt.notna()]
        df.cnt = df.cnt.astype(int)
        df.sort_values('cnt', ascending=False, inplace=True)
        return df

    def get_random(self):
        return pd.read_sql_query(f'SELECT id FROM data ORDER BY RANDOM() LIMIT 1', self.disk_engine).id.iloc[0]


if __name__ == '__main__':
    pass
