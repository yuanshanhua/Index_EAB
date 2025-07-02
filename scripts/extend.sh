uv run heu_run.py --algo extend \
--db_conf_file configuration_loader/database/db_con.conf \
--schema_file configuration_loader/database/schema_job.json \
--exp_conf_file configuration_loader/index_advisor/heu_run_conf/extend_config.json \
--work_file workload_generator/random1000.sql \
--max_indexes 10 \
--max_index_width 5 \
--res_save res.json