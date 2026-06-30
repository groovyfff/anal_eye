.PHONY: rabbitmq-debug rabbitmq-cli

# Probe RabbitMQ from the notification-service container (same network + env).
rabbitmq-debug:
	docker compose exec notification-service python /app/scripts/rabbitmq_debug.py

# Quick broker inspection (management container).
rabbitmq-cli:
	docker compose exec rabbitmq rabbitmqctl list_users
	docker compose exec rabbitmq rabbitmqctl list_vhosts
	docker compose exec rabbitmq rabbitmqctl list_permissions -p analeyes
	docker compose exec rabbitmq rabbitmqctl list_connections name user vhost state
	docker compose exec rabbitmq rabbitmqctl list_consumers -p analeyes
