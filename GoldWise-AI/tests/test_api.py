from fastapi.testclient import TestClient
from backend.main import app
client=TestClient(app)
def test_health():assert client.get('/api/health').json()['status']=='ok'
def test_products():assert len(client.get('/api/products').json())==84
def test_analytics():assert client.get('/api/analytics').json()['sales_transactions']==1000
def test_recommend():assert client.post('/api/recommend',json={'category':'Ring','max_budget_usd':1000}).status_code==200
def test_forecast():assert len(client.get('/api/forecast?days=3').json()['forecast'])==3
