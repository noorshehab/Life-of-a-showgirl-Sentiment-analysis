#Constants
COLUMNS = [
    # Identifiers
    "id",
    "user_id_str",
    "conversation_id_str",

    # Creation time
    "created_at",

    # Author information
    "author/id_str",
    "author/name",
    "author/screen_name",
    "author/location",
    "author/description",
    "author/created_at",
    "author/verified",
    "author/verified_type",
    "author/protected",
    "author/favourites_count",
    "author/followers_count",
    "author/friends_count",
    "author/statuses_count",
    "author/media_count",
    "author/normal_followers_count",
    "author/geo_enabled",
    "author/has_custom_timelines",
    "author/has_extended_profile",
    "author/translator_type_enum",
    "author/is_translator",
    "author/url",
    "author/advertiser_account_type",
    "author/analytics_type",
    "author/can_dm",
    "author/can_media_tag",

    # Tweet content
    "full_text",

    # Conversation control
    "conversation_control/policy",
    "conversation_control/conversation_owner/legacy/screen_name",

    # Reply fields
    "in_reply_to_status_id_str",
    "in_reply_to_user_id_str",
    "in_reply_to_screen_name",

    # Quote-related fields
    "is_quote_status",
    "quoted_status_id_str",

    # Numeric & engagement metrics
    "retweet_count",
    "reply_count",
    "quote_count",
    "favorite_count",
    "bookmark_count",
    "retweeted",
    "favorited",
    "bookmarked",
    
    #search metadata
    "searchTerms",
]
id_cols=[
    "quoted_status_id_str",
    "in_reply_to_status_id_str",
    "in_reply_to_user_id_str",
    "author/id_str","id",
    "user_id_str",
    "conversation_id_str",
]


text_col = "full_text"

CSV_NAMES=['CSVs\dataset_twitter-x-scraper_2025-10-10_10-39-15-248.csv',
'CSVs\dataset_twitter-x-scraper_2025-10-06_20-30-50-967.csv',
'CSVs\dataset_twitter-x-scraper_2025-10-06_21-05-15-811.csv',
'CSVs\dataset_twitter-x-scraper_2025-10-06_21-03-08-626.csv',
'CSVs\dataset_twitter-x-scraper_2025-10-06_21-22-29-835.csv',
'CSVs\dataset_twitter-x-scraper_2025-10-07_17-40-58-412.csv',
'CSVs\dataset_twitter-x-scraper_2025-10-08_11-03-27-968.csv',
'CSVs\dataset_twitter-x-scraper_2025-10-08_11-41-46-517.csv',
'CSVs\dataset_twitter-x-scraper_2025-10-06_19-33-50-822.csv']

SEARCH_TERMS=['life of a showgirl',
              'taylor swift',
              'taylor',
              'new album',
              'album',
              'actually romantic',
              'fate of ophelia',
              'opalite',
              'wood',
              'elizabeth taylor',
              'eldest daughter',
              'wish list',
              'ruin the friendship',
              'father figure',
              'swiftie',
              'cancelled']

DATASET_PATH='CSVs/clean.csv'

PLOTS_PATH='plots'
