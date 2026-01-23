# OpenHCult Ansible

Ubuntu-focused playbook for deploying hcultmon, hcultctrl, and Postgres.

## Usage

1) Copy the inventory and set your host:

```bash
cp ansible/inventory.example.ini ansible/inventory.ini
```

2) Edit `ansible/site.yml` and set `openhcult_repo_url` (or override in
inventory/group vars).

3) Run the playbook:

```bash
ansible-playbook -i ansible/inventory.ini ansible/site.yml
```

The playbook will prompt for the database password.

## Variables

Common defaults live in `ansible/roles/common/defaults/main.yml`.
Postgres defaults live in `ansible/roles/postgres/defaults/main.yml`.

Typical overrides:

```ini
[openhcult:vars]
openhcult_repo_url=https://your.git/openhcult.git
openhcult_repo_version=main
openhcult_database_host=127.0.0.1
openhcult_database_port=5432
openhcult_database_name=hcult
openhcult_database_user=hcult
```

Postgres password is loaded from the environment if not set in vars:

```bash
export OPENHCULT_POSTGRES_PASSWORD='replace-me'
```

Database URL is built from components, with the password loaded from the
environment if not set:

```bash
export OPENHCULT_DATABASE_PASSWORD='replace-me'
```
