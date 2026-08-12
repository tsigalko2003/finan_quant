.PHONY: build demo example test universe live clean

build:
	docker compose build

demo:
	docker compose run --rm qtrends demo --config configs/sample.yaml --output-dir outputs/demo

example:
	bash examples/semiconductors/run.sh

test:
	docker compose run --rm --entrypoint pytest qtrends

universe:
	docker compose run --rm qtrends sync-universe --config configs/semiconductors.yaml

live:
	docker compose run --rm qtrends run --config configs/semiconductors.yaml --output-dir outputs/semiconductors

clean:
	docker compose down --remove-orphans
