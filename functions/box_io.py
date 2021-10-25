import pandas as pd
from io import StringIO
from boxsdk import Client, JWTAuth


def load_beeframe_meta_from_box(meta_name, box_config_file, user_id, folder_id):
    """ Download file like 'img_to_text_df_TOEDIT.csv' from box and load as pandas df.
    Uses the box sdk to download and load direct from server.
    
    Args:
        meta_name: name of csv file. i.e. 'img_to_text_df_TOEDIT.csv'
        box_config_file: full path to local config.json file for box sdk
        user_id: box user_id
        folder_id: folder id for folder that contains the meta file
        
    Returns:
        meta_file loaded as pandas dataframe
    """

    sdk = JWTAuth.from_settings_file(box_config_file)
    client = Client(sdk)
    user = client.user(user_id=user_id).get()
    auth_user = sdk.authenticate_user(user)
    sdk = JWTAuth.from_settings_file(box_config_file, access_token=auth_user)
    client = Client(sdk)

    items = client.folder(folder_id=folder_id).get_items()
    for item in items:
        if item.name == meta_name:
            content = client.file(item.id).content()
            beeframe_meta =  pd.read_csv(StringIO(content.decode("utf-8")))
            return beeframe_meta