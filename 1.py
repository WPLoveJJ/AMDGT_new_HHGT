            # ... (计算完指标后) ...
            
            # 使用格式化字符串强制对齐
            # :<6 表示占用6个字符宽度左对齐，:.5f 表示保留5位小数
            row_str = (
                f"{epoch + 1:<6}"
                f"{time_cost:<8.2f}"
                f"{ll:<10.5f}"
                f"{accuracy:<10.5f}"
                f"{rmse:<10.5f}"
                f"{mae:<10.5f}"
                f"{recall:<10.5f}"
                f"{precision:<10.5f}"
                f"{f1:<10.5f}"
                f"{AUC:<10.5f}"
                f"{AUPR:<10.5f}"
                f"{specificity:<10.5f}"
                f"{brier:<10.5f}"
                f"{tp:<6}"
                f"{fn_count:<6}"
                f"{fp:<6}"
                f"{tn:<6}"
                f"{pos_avg:<10.5f}"
                f"{neg_avg:<10.5f}"
            )
            
            # 直接写入格式化好的字符串，不再需要 join
            log_msg(row_str)

            # ...