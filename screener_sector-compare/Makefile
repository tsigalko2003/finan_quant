.PHONY: build web web-stop universe poc-download poc-analyze poc prod prod-nasdaq test

build:
	docker compose build

web:
	docker compose up -d --build web

web-stop:
	docker compose stop web

universe:
	docker compose run --rm screener refresh-universe --source nasdaq

poc-download:
	docker compose run --rm screener download --stage poc --industry semiconductor

poc-analyze:
	docker compose run --rm screener-offline analyze --stage poc --industry semiconductor

poc: poc-download poc-analyze

prod:
	docker compose run --rm screener download --stage prod --industry semiconductor
	docker compose run --rm screener-offline analyze --stage prod --industry semiconductor

prod-nasdaq: universe
	docker compose run --rm screener download --stage prod --industry nasdaq:semiconductor
	docker compose run --rm screener-offline analyze --stage prod --industry nasdaq:semiconductor

test:
	docker compose run --rm test
