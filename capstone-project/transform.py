import pandas as pd
import re


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def transform_raw_datasets(path):
    df_EU = pd.read_csv(path, 
                        sep=';',
                        encoding='utf8'
                    )
    df_eu_columns = df_EU.columns.to_list()
    df_eu_columns


    df_OFAC = pd.read_csv('datasets/OFAC-financial-sanctions-SDN.csv', 
                        encoding='utf8'
                    )
    df_OFAC_addresses = pd.read_csv('datasets/OFAC-financial-sanctions-addresses.csv', 
                        encoding='utf8'
                    )


    df_ofac_columns = df_OFAC.columns.to_list()
    df_ofac_columns


    df_ofac_columns_add = df_OFAC_addresses.columns.to_list()
    df_ofac_columns_add


    # #### 2. Build single-source-of-truth dataset

    # Build EU name field
    df_EU_names = df_EU.copy()

    df_EU_names["name"] = (
        df_EU_names["NameAlias_WholeName"]
        .fillna(
            df_EU_names["NameAlias_FirstName"].fillna("") + " " +
            df_EU_names["NameAlias_LastName"].fillna("")
        )
    )

    df_EU_names = df_EU_names[
        ["Entity_LogicalId", "name", "Address_CountryIso2Code"]
    ].dropna(subset=["name"])

    df_EU_names["source"] = "EU"
    df_EU_names["raw_source_id"] = df_EU_names["Entity_LogicalId"]


    df_OFAC_names = df_OFAC.copy()

    df_OFAC_names = df_OFAC_names.rename(columns={
        "SDN_Name": "name",
        "ent_num": "raw_source_id"
    })

    df_OFAC_names = df_OFAC_names[
        ["raw_source_id", "name"]
    ]

    df_OFAC_names["source"] = "OFAC"


    df_OFAC_addr = df_OFAC_addresses.rename(columns={
        "ent_num": "raw_source_id",
        "country": "country"
    })

    df_OFAC_names = df_OFAC_names.merge(
        df_OFAC_addr[["raw_source_id", "country"]],
        on="raw_source_id",
        how="left"
    )


    # #### Normalize the names
    def normalize_name(name):
        name = str(name).lower()
        name = re.sub(r"[^a-z0-9\s]", "", name)
        name = re.sub(r"\s+", " ", name).strip()
        return name

    df_EU_names["name_clean"] = df_EU_names["name"].apply(normalize_name)
    df_OFAC_names["name_clean"] = df_OFAC_names["name"].apply(normalize_name)



    # #### Unify datasets
    sanctions_df = pd.concat([
        df_EU_names[["raw_source_id", "source", "name", "name_clean", "Address_CountryIso2Code"]],
        df_OFAC_names[["raw_source_id", "source", "name", "name_clean", "country"]]
    ], ignore_index=True)

    sanctions_df = sanctions_df.drop_duplicates(subset=["source", "name_clean"])



    sanctions_df = sanctions_df.rename(columns={
        "Address_CountryIso2Code": "country_iso2"
    })

    return sanctions_df




