import pandas as pd
import numpy as np
from time import time
from sqlalchemy import create_engine  # database connection


def create_db(disk_engine, group_n: int = 3):
    chunksize = 20000
    n = 4126183
    chunks = int(n / chunksize)
    i = 1
    t1 = time()
    for df in pd.read_csv('data/raw/NABBP_2020_grp_{:02d}.csv'.format(group_n),
                          usecols=[
                              'BAND',
                              'LAT_DD',
                              'LON_DD',
                              'EVENT_DATE',
                              'SPECIES_ID',
                              'EVENT_TYPE',
                              'RECORD_SOURCE'
                          ],
                          parse_dates=['EVENT_DATE'],
                          date_parser=lambda x: pd.to_datetime(x,
                                                               format='%m/%d/%Y',
                                                               errors="coerce"),
                          chunksize=chunksize,
                          iterator=True
                          ):
        df = df[df['LAT_DD'].notna()]
        df = df[df['EVENT_DATE'].notna()]
        df.dropna(axis=1, inplace=True)
        # df.drop(['EVENT_DAY', 'EVENT_MONTH', 'EVENT_YEAR', 'ORIGINAL_BAND', 'BAND_TYPE_CODE', 'ISO_COUNTRY', 'COORDINATES_PRECISION_CODE'], axis=1, inplace=True)
        df.SPECIES_ID = df.SPECIES_ID.astype(np.uint16)

        df.RECORD_SOURCE = df.RECORD_SOURCE.astype('category')
        df.EVENT_TYPE = df.EVENT_TYPE.astype('category')
        df.rename({
            'BAND': 'band',
            'LAT_DD': 'lat',
            'LON_DD': 'lon',
            'EVENT_DATE': 't',
            'SPECIES_ID': 'id',
            'EVENT_TYPE': 'event',
            'RECORD_SOURCE': 'source'
        }, axis=1, inplace=True)
        if i % 10 == 0:
            print(
                f'{int(i / chunks * 100)}% -- '
                f'({int(time() - t1)}s)', end='\r'
            )
        df.to_sql('data', disk_engine, if_exists='append')
        i += 1


def main():
    gp_n = 3  # Choose Dataset to load

    create_db(create_engine('sqlite:///data/db/bbp_{}.db'.format(gp_n)), group_n=gp_n)


if __name__ == '__main__':
    main()
