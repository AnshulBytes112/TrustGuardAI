install:
	pip install -e ".[dev]"
	cd frontend && npm install

test:
	pytest tests/backend tests/ml

test-frontend:
	cd frontend && npm run build

lint:
	ruff check .

format:
	black .

backend:
	uvicorn backend.main:app --reload

frontend:
	cd frontend && npm run dev

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down
