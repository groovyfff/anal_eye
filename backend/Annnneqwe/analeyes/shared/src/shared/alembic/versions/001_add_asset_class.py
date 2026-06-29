"""add asset_class and relax crypto-only columns

Revision ID: 001
Revises:
Create Date: 2026-06-25

Compatible with fresh installs where tables are created later via OrmBase.metadata.create_all:
if a table does not exist yet, column changes are skipped so create_all can provision the full schema.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision = '001'
down_revision = None
branch_labels = None
depends_on = None

TABLES_WITH_ASSET_CLASS = (
    'signal_feature_logs',
    'ensemble_model_decisions',
    'ensemble_backtest_results',
    'trades',
)

CRYPTO_ONLY_NULLABLE_COLUMNS: dict[str, list[str]] = {
    'signal_feature_logs': [
        'feat_funding_rate',
        'feat_open_interest_z',
        'feat_liquidations_long_usd',
        'feat_liquidations_short_usd',
        'feat_cvd',
    ],
}


def _table_exists(inspector: inspect, table_name: str) -> bool:
    return table_name in inspector.get_table_names()


def _column_names(inspector: inspect, table_name: str) -> set[str]:
    return {column['name'] for column in inspector.get_columns(table_name)}


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)

    for table_name in TABLES_WITH_ASSET_CLASS:
        if not _table_exists(inspector, table_name):
            continue
        columns = _column_names(inspector, table_name)
        if 'asset_class' in columns:
            continue
        op.add_column(
            table_name,
            sa.Column(
                'asset_class',
                sa.String(length=16),
                nullable=False,
                server_default='crypto',
            ),
        )

    for table_name, column_names in CRYPTO_ONLY_NULLABLE_COLUMNS.items():
        if not _table_exists(inspector, table_name):
            continue
        existing_columns = _column_names(inspector, table_name)
        for column_name in column_names:
            if column_name not in existing_columns:
                continue
            op.alter_column(
                table_name,
                column_name,
                existing_type=sa.Float(),
                nullable=True,
            )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)

    for table_name in TABLES_WITH_ASSET_CLASS:
        if not _table_exists(inspector, table_name):
            continue
        if 'asset_class' not in _column_names(inspector, table_name):
            continue
        op.drop_column(table_name, 'asset_class')
