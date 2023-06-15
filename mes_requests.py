import requests, json

def post_inspection(inspection_json):
    try:
        url = f"https://mj.d-triple.com/api/mes/v1/external/inspection"
        
        headers = {"Content-Type": "application/json"}
        print(inspection_json)
        response = requests.post(url, headers=headers, json=inspection_json)
        return response.status_code
    except:
        return 400
        print("error")


def get_inspector():
    try:
        url = "https://mj.d-triple.com/api/mes/v1/external/inspector"
        headers = {"Content-Type": "application/json"}
        response = requests.get(url, headers=headers)
    
        return json.loads(response.text.encode().decode('unicode-escape'))["objects"]

    except:
        return []

def get_workorder():
    try:
        url = "https://mj.d-triple.com/api/mes/v1/external/workorder"
        headers = {"Content-Type": "application/json"}
        response = requests.get(url, headers=headers)
    
        return json.loads(response.text.encode().decode('unicode-escape'))["objects"]
    except:
        return []

def get_workorder_list(id):
    try:
        url = f"https://mj.d-triple.com/api/mes/v1/external/workorder/{id}"
        headers = {"Content-Type": "application/json"}
        response = requests.get(url, headers=headers)

        return json.loads(response.text.encode().decode('unicode-escape'))["objects"]
    except:
        return []