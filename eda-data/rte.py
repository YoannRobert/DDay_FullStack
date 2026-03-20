import pandas as pd
import requests
import os

url_token_oauth = "https://digital.iservices.rte-france.com/token/oauth/"
headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": "Basic " + os.getenv("RTE_SECRET_KEY")
    }
response_token = requests.post(url_token_oauth, headers=headers)
data = response_token.json()
token_oauth = data["access_token"]

start_date = '2026-01-15T00:00:00+00:00'
end_date = '2026-03-20T09:00:00+00:00'
type = 'REALISED'

api_url = "https://digital.iservices.rte-france.com/open_api/consumption/v1/short_term"

headers = {
    "Authorization": "Bearer " + token_oauth
}

params = {
    "type": type,
    "start_date": start_date,
    "end_date": end_date
}

response = requests.get(api_url, headers=headers, params=params)
data = response.json()

df_conso = pd.DataFrame({'start_date': [], 'end_date': [], 'updated_date': [], 'value': []})

for i in range(1, 3361): # 3360 1/4 d'heures = 35 jours
    observation = pd.DataFrame(data['short_term'][0]['values'][-i], index=[0])
    df_conso = pd.concat([df_conso, observation], ignore_index=True)