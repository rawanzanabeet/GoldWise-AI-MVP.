"""GoldWise AI MVP: synthetic-data retail intelligence and image search."""
from pathlib import Path
from io import BytesIO
import csv
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import cosine_similarity

ROOT=Path(__file__).resolve().parents[1]
PRODUCTS=ROOT/'data/products.csv'
SALES=ROOT/'data/sales_1000.csv'
app=FastAPI(title='GoldWise AI',version='0.1.0',description='Educational MVP; sample data is synthetic.')
app.add_middleware(CORSMiddleware,allow_origins=['*'],allow_methods=['GET','POST'],allow_headers=['*'])

def tables():
    return pd.read_csv(PRODUCTS).fillna(''),pd.read_csv(SALES,parse_dates=['date'])

def records(df):
    return df.replace({np.nan:None}).to_dict(orient='records')

@app.get('/api/health')
def health():return {'status':'ok','dataset':'synthetic','version':'0.1.0'}

@app.get('/api/products')
def products(category:str|None=None,karat:int|None=None,style:str|None=None,search:str|None=None):
    df,_=tables()
    if category:df=df[df.category.str.lower()==category.lower()]
    if karat:df=df[df.karat==karat]
    if style:df=df[df['style'].str.lower()==style.lower()]
    if search:df=df[df.name.str.contains(search,case=False,regex=False)]
    return records(df)

@app.get('/api/analytics')
def analytics():
    products,sales=tables(); merged=sales.merge(products,left_on='product_id',right_on='id')
    daily=sales.assign(day=sales.date.dt.strftime('%Y-%m-%d')).groupby('day',as_index=False).revenue_usd.sum()
    by_category=merged.groupby('category',as_index=False).agg(revenue_usd=('revenue_usd','sum'),units=('quantity','sum'))
    sold=sales.groupby('product_id').quantity.sum().to_dict()
    products['units_sold']=products.id.map(sold).fillna(0).astype(int)
    slow=products[(products.stock>0)].sort_values(['units_sold','stock'],ascending=[True,False]).head(10)
    return {'total_revenue_usd':round(float(sales.revenue_usd.sum()),2),'sales_transactions':len(sales),'products':len(products),'low_stock':int((products.stock<=5).sum()),'category_performance':records(by_category),'daily_revenue':records(daily),'slow_moving':records(slow[['id','name','stock','units_sold']]),'top_products':records(products.sort_values('units_sold',ascending=False).head(8)[['id','name','units_sold']]),'disclaimer':'Synthetic demonstration data; not actual store sales.'}

@app.get('/api/forecast')
def forecast(days:int=14):
    if not 1<=days<=90:raise HTTPException(400,'days must be between 1 and 90')
    _,sales=tables(); daily=sales.groupby('date').quantity.sum().asfreq('D',fill_value=0)
    # Date features and lag features; forecast recursively after the observed period.
    series=daily.astype(float).tolist(); dates=daily.index
    X=[];y=[]
    for i in range(14,len(series)):
        dt=dates[i];X.append([dt.dayofweek,dt.month,dt.dayofyear,*series[i-7:i],np.mean(series[i-14:i])]);y.append(series[i])
    if len(X)<30:raise HTTPException(400,'Not enough historical data')
    model=RandomForestRegressor(n_estimators=80,max_depth=9,random_state=42,n_jobs=1)
    model.fit(X,y)
    output=[]
    for j in range(days):
        dt=dates[-1]+pd.Timedelta(days=j+1)
        features=[dt.dayofweek,dt.month,dt.dayofyear,*series[-7:],np.mean(series[-14:])]
        pred=max(0,round(float(model.predict([features])[0])))
        output.append({'date':dt.strftime('%Y-%m-%d'),'predicted_units':pred});series.append(pred)
    return {'method':'RandomForestRegressor with calendar and lag features','forecast':output,'disclaimer':'Demo forecast trained on synthetic data; not validated for business decisions.'}

class Preferences(BaseModel):
    category:str|None=None
    karat:int|None=None
    style:str|None=None
    max_budget_usd:float|None=Field(default=None,ge=0)
    max_weight_g:float|None=Field(default=None,gt=0)
    in_stock_only:bool=True

@app.post('/api/recommend')
def recommend(prefs:Preferences):
    df,_=tables()
    if prefs.in_stock_only:df=df[df.stock>0]
    if prefs.max_budget_usd is not None:df=df[df.price_usd<=prefs.max_budget_usd]
    if prefs.max_weight_g is not None:df=df[df.weight_g<=prefs.max_weight_g]
    if prefs.karat is not None:df=df[df.karat==prefs.karat]
    if prefs.category:df=df[df.category.str.lower()==prefs.category.lower()]
    if prefs.style:df=df[df['style'].str.lower()==prefs.style.lower()]
    if df.empty:return {'results':[],'explanation':'No catalog items match all filters.'}
    # Transparent content-based ranking, not learned customer personalization.
    df=df.copy();df['rank_score']=df.stock.clip(upper=20)/20 + (1/(1+df.price_usd/1000))
    return {'results':records(df.sort_values('rank_score',ascending=False).head(12)),'explanation':'Rule-based product filtering and availability/price ranking; no customer model trained.'}

def image_features(img):
    img=ImageOps.exif_transpose(img).convert('RGB').resize((128,128))
    arr=np.asarray(img)
    # Color histogram is a basic computer-vision baseline, not semantic image recognition.
    hist=np.concatenate([np.histogram(arr[:,:,c],bins=16,range=(0,256))[0] for c in range(3)]).astype(float)
    return hist/(np.linalg.norm(hist)+1e-10)

@app.post('/api/visual-search')
async def visual_search(file:UploadFile=File(...)):
    data=await file.read(5_000_001)
    if len(data)>5_000_000:raise HTTPException(413,'Image must be 5MB or smaller')
    try:query=image_features(Image.open(BytesIO(data)))
    except Exception:raise HTTPException(400,'Please upload a valid image')
    df,_=tables();candidates=[]
    for row in df.to_dict('records'):
        # Only locally stored catalog images are searched; the synthetic catalog has none.
        path=ROOT/'data/catalog_images'/f"{int(row['id'])}.jpg"
        if not path.exists():continue
        try:
            with Image.open(path) as im: score=float(np.dot(query,image_features(im)))
            candidates.append({'product_id':int(row['id']),'name':row['name'],'similarity':round(score,4)})
        except Exception:continue
    candidates.sort(key=lambda x:x['similarity'],reverse=True)
    return {'results':candidates[:10],'indexed_images':len(candidates),'method':'RGB histogram cosine similarity baseline','note':'Add real product images as data/catalog_images/<product_id>.jpg to enable matching. No semantic vision model is included.'}

app.mount('/',StaticFiles(directory=ROOT/'frontend',html=True),name='frontend')
