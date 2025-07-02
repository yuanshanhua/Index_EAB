# -*- coding: utf-8 -*-
# @Project: index_eab
# @Module: heu_run
# @Author: Wei Zhou
# @Time: 2023/8/16 14:42

import configparser
import json

from index_advisor_selector.index_selection.heu_selection.heu_algos.anytime_algorithm import (
    AnytimeAlgorithm,
)
from index_advisor_selector.index_selection.heu_selection.heu_algos.auto_admin_algorithm import (
    AutoAdminAlgorithm,
)
from index_advisor_selector.index_selection.heu_selection.heu_algos.cophy_algorithm import (
    CoPhyAlgorithm,
)
from index_advisor_selector.index_selection.heu_selection.heu_algos.db2advis_algorithm import (
    DB2AdvisAlgorithm,
    IndexBenefit,
)
from index_advisor_selector.index_selection.heu_selection.heu_algos.drop_algorithm import (
    DropAlgorithm,
)
from index_advisor_selector.index_selection.heu_selection.heu_algos.extend_algorithm import (
    ExtendAlgorithm,
)
from index_advisor_selector.index_selection.heu_selection.heu_algos.relaxation_algorithm import (
    RelaxationAlgorithm,
)
from index_advisor_selector.index_selection.heu_selection.heu_utils import heu_com
from index_advisor_selector.index_selection.heu_selection.heu_utils.heu_com import (
    get_parser,
)
from index_advisor_selector.index_selection.heu_selection.heu_utils.index import Index
from index_advisor_selector.index_selection.heu_selection.heu_utils.postgres_dbms import (
    PostgresDatabaseConnector,
)
from index_advisor_selector.index_selection.heu_selection.heu_utils.workload import (
    Workload,
)

ALGORITHMS = {
    "auto_admin": AutoAdminAlgorithm,
    "db2advis": DB2AdvisAlgorithm,
    "drop": DropAlgorithm,
    "extend": ExtendAlgorithm,
    "relaxation": RelaxationAlgorithm,
    "anytime": AnytimeAlgorithm,
    "cophy": CoPhyAlgorithm,
}


class IndexEncoder(json.JSONEncoder):
    def default(self, obj):
        # 👇️ if passed in object is instance of Decimal
        # convert it to a string

        # db2advis
        if isinstance(obj, IndexBenefit):
            return str(obj)
        if isinstance(obj, Index):
            return str(obj)
        if "Workload" in str(obj.__class__):
            return str(obj)
        if "Index" in str(obj.__class__):
            return str(obj)
        if "Column" in str(obj.__class__):
            return str(obj)
        if isinstance(obj, set):
            return list(obj)

        # 👇️ otherwise use the default behavior
        return json.JSONEncoder.default(self, obj)


def get_heu_result(args, algos, work_list):
    print(f"开始处理算法: {algos}")
    print(
        f"工作负载包含 {len(work_list) if hasattr(work_list, '__len__') else 'N/A'} 个查询"
    )

    print("读取数据库配置文件...")
    db_conf = configparser.ConfigParser()
    db_conf.read(args.db_conf_file)

    print(f"从模式文件获取列信息: {args.schema_file}")
    _, columns = heu_com.get_columns_from_schema(args.schema_file)
    print(f"获取到 {len(columns)} 个列")

    print("建立数据库连接...")
    connector = PostgresDatabaseConnector(
        db_conf,
        autocommit=True,
        host=args.host,
        port=args.port,
        db_name=args.db_name,
        user=args.user,
        password=args.password,
    )
    print("数据库连接建立成功")

    res_data = dict()
    for algo in algos:
        print(f"开始处理算法: {algo}")
        # indexes, no_cost, total_no_cost, ind_cost, total_ind_cost, sel_info
        exp_conf_file = args.exp_conf_file.format(algo)
        print(f"读取实验配置文件: {exp_conf_file}")

        with open(exp_conf_file, "r") as rf:
            exp_config = json.load(rf)

        configs = heu_com.find_parameter_list(
            exp_config["algorithms"][0], params=args.sel_params
        )
        print(f"找到 {len(configs)} 个配置")

        # (0824): newly modified.
        print("创建工作负载对象...")
        workload = Workload(
            heu_com.read_row_query(
                work_list,
                exp_config,
                columns,
                type="",
                varying_frequencies=args.varying_frequencies,
                seed=args.seed,
            )
        )
        print(f"工作负载包含 {len(workload.queries)} 个查询")

        data = list()
        for config_idx, config in enumerate(configs, 1):
            print(f"处理配置 {config_idx}/{len(configs)}: {config['parameters']}")

            print("删除假设索引...")
            connector.drop_hypo_indexes()

            # (0818): newly added.
            if args.constraint is not None:
                config["parameters"]["constraint"] = args.constraint
                print(f"设置约束条件: {args.constraint}")
            if args.budget_MB is not None:
                config["parameters"]["budget_MB"] = args.budget_MB
                print(f"设置存储预算: {args.budget_MB} MB")
            if args.max_indexes is not None:
                config["parameters"]["max_indexes"] = args.max_indexes
                print(f"设置最大索引数: {args.max_indexes}")

            # (0926): newly added.
            if "max_index_width" in args and args.max_index_width is not None:
                config["parameters"]["max_index_width"] = args.max_index_width
                print(f"设置最大索引宽度: {args.max_index_width}")

            # (0918): newly added.
            if algo == "drop" and "multi_column" in args:
                config["parameters"]["multi_column"] = args.multi_column
                print(f"设置多列索引: {args.multi_column}")

            # (1211): newly added. for `cophy`
            if algo == "cophy":
                config["parameters"]["ampl_bin_path"] = args.ampl_bin_path
                config["parameters"]["ampl_mod_path"] = args.ampl_mod_path
                config["parameters"]["ampl_dat_path"] = args.ampl_dat_path
                config["parameters"]["ampl_solver"] = args.ampl_solver
                print("配置CoPhY算法的AMPL参数")

            print(f"初始化{algo}算法...")
            algorithm = ALGORITHMS[algo](
                connector,
                config["parameters"],
                args.process,
                args.cand_gen,
                args.is_utilized,
                args.sel_oracle,
            )

            # return algorithm.get_index_candidates(workload, db_conf=db_conf, columns=columns)

            print("计算最佳索引...")
            if not args.process and not args.overhead:
                sel_info = ""
                indexes = algorithm.calculate_best_indexes(
                    workload, overhead=args.overhead, db_conf=db_conf, columns=columns
                )
            else:
                indexes, sel_info = algorithm.calculate_best_indexes(
                    workload, overhead=args.overhead, db_conf=db_conf, columns=columns
                )

            print(f"找到 {len(indexes)} 个推荐索引")

            indexes = [str(ind) for ind in indexes]
            cols = [ind.split(",") for ind in indexes]
            cols = [list(map(lambda x: x.split(".")[-1], col)) for col in cols]
            indexes = [
                f"{ind.split('.')[0]}#{','.join(col)}"
                for ind, col in zip(indexes, cols)
            ]
            indexes_res = [
                f"{ind.split('.')[0]}({','.join(col)})"
                for ind, col in zip(indexes, cols)
            ]

            if indexes:
                print(f"推荐的索引: {indexes}")
            else:
                print("未找到推荐索引")

            no_cost, ind_cost = list(), list()
            total_no_cost, total_ind_cost = 0, 0

            # # (0916): newly added.
            # freq_list = [1 for _ in work_list]
            # if isinstance(work_list[0], list):
            #     work_list = [item[1] for item in work_list]
            #     if args.varying_frequencies:
            #         freq_list = [item[-1] for item in work_list]
            #
            # # (0916): newly modified.
            # for sql, freq in zip(work_list, freq_list):
            #     no_cost_ = connector.get_ind_cost(sql, "") * freq
            #     total_no_cost += no_cost_
            #     no_cost.append(no_cost_)
            #
            #     ind_cost_ = connector.get_ind_cost(sql, indexes) * freq
            #     total_ind_cost += ind_cost_
            #     ind_cost.append(ind_cost_)

            # (0916): newly modified.
            print("计算查询成本...")
            freq_list = list()
            for query_idx, query in enumerate(workload.queries, 1):
                if query_idx % 10 == 0 or query_idx == len(workload.queries):
                    print(f"正在计算查询 {query_idx}/{len(workload.queries)} 的成本")

                no_cost_ = connector.get_ind_cost(query.text, "") * query.frequency
                total_no_cost += no_cost_
                no_cost.append(no_cost_)

                ind_cost_ = (
                    connector.get_ind_cost(query.text, indexes) * query.frequency
                )
                total_ind_cost += ind_cost_
                ind_cost.append(ind_cost_)

                freq_list.append(query.frequency)

            cost_reduction = total_no_cost - total_ind_cost
            cost_reduction_pct = (
                (cost_reduction / total_no_cost * 100) if total_no_cost > 0 else 0
            )
            print(
                f"成本分析完成 - 无索引总成本: {total_no_cost:.2f}, 有索引总成本: {total_ind_cost:.2f}"
            )
            print(f"成本降低: {cost_reduction:.2f} ({cost_reduction_pct:.2f}%)")

            # (0916): newly added.
            if args.varying_frequencies:
                data.append(
                    {
                        "config": config["parameters"],
                        "workload": [work_list, freq_list],
                        "indexes": indexes_res,
                        "no_cost": no_cost,
                        "total_no_cost": total_no_cost,
                        "ind_cost": ind_cost,
                        "total_ind_cost": total_ind_cost,
                        "sel_info": sel_info,
                    }
                )
            else:
                data.append(
                    {
                        "config": config["parameters"],
                        "workload": work_list,
                        "indexes": indexes_res,
                        "no_cost": no_cost,
                        "total_no_cost": total_no_cost,
                        "ind_cost": ind_cost,
                        "total_ind_cost": total_ind_cost,
                        "sel_info": sel_info,
                    }
                )

        if len(data) == 1:
            data = data[0]

        res_data[algo] = data
        print(f"算法 {algo} 处理完成")

    print("所有算法处理完成")
    return res_data


if __name__ == "__main__":
    print("=== 启动索引建议器程序 ===")

    parser = get_parser()
    args = parser.parse_args()

    algos = [args.algo] if args.algo else ALGORITHMS.keys()
    print(f"使用算法: {algos}")

    args.constraint = "storage"
    args.budget_MB = 500
    print(f"约束条件: {args.constraint}, 存储预算: {args.budget_MB} MB")

    # args.constraint = "number"
    args.max_indexes = 5
    print(f"最大索引数: {args.max_indexes}")

    args.multi_column = True
    print(f"支持多列索引: {args.multi_column}")

    print(f"读取工作负载文件: {args.work_file}")
    work_list = None
    if args.work_file.endswith(".sql"):
        with open(args.work_file, "r") as rf:
            work_list = rf.readlines()
        print(f"从SQL文件读取了 {len(work_list)} 行查询")
    elif args.work_file.endswith(".json"):
        with open(args.work_file, "r") as rf:
            work_list = json.load(rf)
        print(f"从JSON文件读取了 {len(work_list)} 个工作负载")

    if work_list is None:
        print("未能读取工作负载文件")
        exit(1)

    datas = []
    for work_idx, work in enumerate(work_list, 1):
        print(f"=== 处理工作负载 {work_idx}/{len(work_list)} ===")
        print(f"工作负载内容: {work}")
        if isinstance(work, str):
            work = [work]
        data = get_heu_result(args, algos, work)
        print(f"工作负载 {work_idx} 处理完成")
        print(f"结果: {data}")
        datas.append(data)

    print(f"所有 {len(work_list)} 个工作负载处理完成")

    if args.res_save is not None:
        print(f"保存结果到文件: {args.res_save}")
        with open(args.res_save, "w") as wf:
            json.dump(datas, wf, indent=2, cls=IndexEncoder)
        print("结果保存完成")
    else:
        print("未指定输出文件，结果未保存")

    print("=== 程序执行完成 ===")
