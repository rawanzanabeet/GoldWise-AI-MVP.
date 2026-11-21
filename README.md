GoldWise AI — Jewelry Retail Intelligence MVP
A runnable early-stage prototype combining jewelry inventory analytics, data mining, basic recommendation ranking, machine-learning demand forecasting, and image-based catalog matching.
Integrity notice: The 1,000 sales records and 84 products are synthetically generated. They are not real store transactions. This repository does not claim validated demand predictions, real customer deployments, trained semantic jewelry recognition, or commercial readiness.

Features implemented
Feature	Implementation	Status
Dashboard	Revenue, sales, category performance, slow-moving inventory	Working (synthetic data)
Catalog	Search/filter product data	Working
Recommendations	Preferences: type, karat, style, budget, weight; stock/price ranking	Working (rule-based)
Demand forecasting	RandomForestRegressor with lag and calendar features	Working demo; not validated
Visual search	RGB color histogram + cosine similarity on local catalog photos	Working baseline when photos supplied
Image processing	PIL resizing, orientation correction, RGB conversion	Working
Data mining	Aggregations, slow-moving product identification	Working
User authentication	Not included	Future
Multi-store tenancy	Not included	Future
Deep visual embeddings / semantic search	Not included	Future
Customer-trained personalized model	Not included	Future
Production forecasting validation	Not included	Future


Quick start
Requires Python 3.10+.
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload
Open http://127.0.0.1:8000 for the dashboard and http://127.0.0.1:8000/docs for the API documentation.
Visual search setup
Place product images in data/catalog_images/ using their catalog IDs, e.g. 1.jpg, 2.jpg, 3.jpg. Upload a query photo through the Visual Search tab. The algorithm uses basic color histograms and can miss shape/style similarities; pretrained vision embeddings are a future upgrade. Images are not bundled because they require licensing and store permission.
Dataset
- data/products.csv: 84 synthetic products, with category, style, karat, weight, price and stock.
- data/sales_1000.csv: exactly 1,000 synthetic transactions.
No customer-identifying information is included.
Research and evidence
- Exploratory survey
- Interactive earlier concept prototype
Survey responses and outreach screenshots are intentionally not mirrored into this public repository because they may contain personal data or require permission. Survey evidence reflects a small convenience sample and should not be generalized to the market.
Suggested next steps
1. Validate forecasting with time-based holdout and MAE/WAPE metrics on real permissioned sales data.
2. Add CLIP-style visual embeddings and FAISS retrieval using permissioned jewelry catalog photos.
3. Add accounts, RBAC, store-specific isolation, and audit logs before onboarding stores.
4. Validate recommendations and willingness to pay with real retailers.
5. Add CSV import validation, PostgreSQL persistence, and secure deployment.
Author
Rawan Zanabeet — GoldWise AI concept and prototype.
