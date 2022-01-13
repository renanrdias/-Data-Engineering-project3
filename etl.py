import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, dayofweek, monotonically_increasing_id


config = configparser.ConfigParser()
config.read_file(open('dl.cfg'))


os.environ['AWS_ACCESS_KEY_ID'] = config['AWS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY'] = config['AWS']['AWS_SECRET_ACCESS_KEY']
os.environ['AWS_DEFAULT_REGION'] = config['AWS']['AWS_DEFAULT_REGION']


def create_spark_session():
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.10.1") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    """
    Loads and processes the song data stored in a S3 bucket and then saves it in another S3 bucket
    in parquet format.
    It creates and saves dimension tables as follows:
    - songs_table
    - artists_table

    spark: SparkSession
    input_data: The S3 bucket where the song data json files are stored.
    output_data: The S3 bucket where processed data will be stored.

    """
    # get filepath to song data file
    song_data = os.path.join(input_data, 'song_data/*/*/*/*.json')

    # read song data file
    df = spark.read.json(song_data)

    # extract columns to create songs table
    songs_table = df.select('song_id', 'title', 'artist_id', 'year', 'duration')\
        .drop_duplicates(['song_id'])

    # write songs table to parquet files partitioned by year and artist
    songs_table.write.partitionBy('year', 'artist_id')\
        .mode('overwrite')\
        .parquet(os.path.join(output_data, 'songs/songs.parquet'))

    # extract columns to create artists table
    artists_table = df.select('artist_id', 'artist_name', 'artist_location', 'artist_latitude', 'artist_longitude')\
        .drop_duplicates(['artist_id'])

    # write artists table to parquet files
    artists_table.write.mode('overwrite').parquet(os.path.join(output_data, 'artists/artists.parquet'))


def process_log_data(spark, input_data, output_data):
    """
    Loads and processes the log metadata stored in a S3 bucket and then saves it into another S3 bucket
    in parquet format.
    It creates and saves dimension tables as follows:
    - users_table
    - time_table
    It also creates a fact table as follows:
    - songplays_table

    spark: SparkSession
    input_data: The S3 bucket where the log data json files are stored.
    output_data: The S3 bucket where processed data will be stored.

    """
    # get filepath to log data file
    log_data = os.path.join(input_data, 'log_data/*/*/*.json')

    # read log data file
    df = spark.read.json(log_data)

    # filter by actions for song plays
    df = df.filter(df['page'] == 'NextSong')

    # extract columns for users table
    users_table = df.select('userId', 'firstName', 'lastName', 'gender', 'level').drop_duplicates(['userId'])

    # write users table to parquet files
    users_table.write.mode('overwrite').parquet(os.path.join(output_data, 'users/users.parquet'))

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: str(int(int(x)/1000)))
    df = df.withColumn('timestamp', get_timestamp(df.ts))

    # create datetime column from original timestamp column
    get_datetime = udf(lambda x: str(datetime.fromtimestamp(int(x))))
    df = df.withColumn('datetime', get_datetime(df.timestamp))

    # extract columns to create time table
    time_table = df.select('datetime') \
        .withColumn('start_time', df.datetime) \
        .withColumn('hour', hour(df.datetime)) \
        .withColumn('day', dayofmonth(df.datetime)) \
        .withColumn('week', weekofyear(df.datetime)) \
        .withColumn('month', month(df.datetime)) \
        .withColumn('year', year(df.datetime)) \
        .withColumn('weekday', dayofweek(df.datetime)) \
        .select('start_time', 'hour', 'day', 'week', 'month', 'year', 'weekday')\
        .drop_duplicates(['start_time']).orderBy('start_time')

    # write time table to parquet files partitioned by year and month
    time_table.write.partitionBy('year', 'month').mode('overwrite')\
        .parquet(os.path.join(output_data, 'times/time_data.parquet'))

    # read in song data to use for songplays table
    song_data = os.path.join(input_data, 'song_data/*/*/*/*.json')
    song_df = spark.read.json(song_data)

    # extract columns from joined song and log datasets to create songplays table
    songplays_table = df\
        .join(song_df, [df.song == song_df.title, df.artist == song_df.artist_name, df.length == song_df.duration])\
        .drop_duplicates()\
        .withColumn('songplay_id', monotonically_increasing_id()) \
            .select(
                col('songplay_id'),
                col('datetime').alias('start_time'),
                col('userId').alias('user_id'),
                col('level'),
                col('song_id'),
                col('artist_id'),
                col('sessionId').alias('session_id'),
                col('location'),
                col('userAgent').alias('user_agent'),
                col('year'),
                month('datetime').alias('month'))

    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.partitionBy('year', 'month').mode('overwrite')\
        .parquet(os.path.join(output_data, 'songplays/songplays.parquet'))


def main():
    spark = create_spark_session()

    input_data = 's3a://udacity-dend/'
    output_data = 's3a://spark-bucket-rrmd/project-files/'

    process_song_data(spark, input_data, output_data)

    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()

