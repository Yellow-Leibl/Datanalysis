from Datanalysis import (
    SamplingData, DoubleSampleData, SamplingDatas, TimeSeriesData)
from Datanalysis.SamplesTools import formRowNV, PROTOCOL_TITLE

import Datanalysis.functions as func


class ProtocolGenerator:
    @staticmethod
    def getProtocol(data):
        if isinstance(data, SamplingData):
            return ProtocolGenerator.get_for_sampling_data(data)
        elif isinstance(data, DoubleSampleData):
            return ProtocolGenerator.get_for_double_sample(data)
        elif isinstance(data, SamplingDatas):
            return ProtocolGenerator.get_for_sampling_datas(data)
        elif isinstance(data, TimeSeriesData):
            return ProtocolGenerator.get_protocol_time_series(data)
        else:
            raise TypeError("data is not a valid type")

    @staticmethod
    def get_for_sampling_data(data: SamplingData):
        inf_protocol = []
        def add_text(text=""): inf_protocol.append(text)
        def addForm(title, *args): inf_protocol.append(formRowNV(title, *args))

        add_text(PROTOCOL_TITLE)
        addForm('Характеристика', 'INF', 'Значення', 'SUP', 'SKV')
        add_text()

        addForm("Сер арифметичне", data.x_-data.det_x_, data.x_,
                data.x_+data.det_x_, data.det_x_)

        addForm("Дисперсія", data.S-data.det_S, data.S, data.S+data.det_S,
                data.det_S)

        addForm("Сер квадратичне", data.Sigma-data.det_Sigma, data.Sigma,
                data.Sigma+data.det_Sigma, data.det_Sigma)

        addForm("Коеф. асиметрії", data.A-data.det_A, data.A,
                data.A+data.det_A, data.det_A)

        addForm("Коеф. ексцесу", data.E-data.det_E, data.E,
                data.E+data.det_E, data.det_E)

        addForm("Коеф. контрексцесу", data.c_E-data.det_c_E, data.c_E,
                data.c_E+data.det_c_E, data.det_c_E)

        addForm("Коеф. варіації Пірсона", data.W_-data.det_W_, data.W_,
                data.W_+data.det_W_, data.det_W_)

        add_text()

        addForm("MED", data.MED)
        addForm("усіченне середнє", data.x_a)
        addForm("MED Уолша", data.MED_Walsh)
        addForm("MAD", data.MAD)
        addForm("непарам. коеф. варіацій", data.Wp)
        addForm("коеф. інтер. розмаху", data.inter_range)

        add_text()

        addForm("мат спод.інтерв.передбачення", data.x_ - data.vanga_x_, "",
                data.x_ + data.vanga_x_)

        add_text()

        addForm("Квантилі", "Ймовірність", "X")
        for i in range(len(data.quant)):
            step_quant = 1 / len(data.quant)
            addForm("", f"{step_quant * (i + 1):.3}", data.quant[i])

        return "\n".join(inf_protocol)

    def kolmogorov_test(self, data: SamplingData, f_repr):
        res = data.kolmogorov_test(f_repr)
        crits = f"{self.kolmogorov_pz:.5} >= {self.kolmogorov_alpha_zgodi:.5}"
        if res:
            return "Відтворення адекватне за критерієм" \
                f" згоди Колмогорова: {crits}"
        else:
            return "Відтворення неадекватне за критерієм" \
                f" згоди Колмогорова: {crits}"

    def xixi_test_1d(data: SamplingData, hist_class, f):
        res = data.xiXiTest(f, hist_class)
        crits = f"{data.xixitest_x_2:.5} < {data.xixitest_quant:.5}"
        if res:
            return "Відтворення адекватне за критерієм" \
                f" Пірсона: {crits}"
        else:
            return "Відтворення неадекватне за критерієм" \
                f" Пірсона: {crits}"

#
#
#
#
#
#
#

    @staticmethod
    def get_for_double_sample(data: DoubleSampleData):
        inf_protocol = []
        def add_text(text=""): inf_protocol.append(text)
        def addForm(title, *args): inf_protocol.append(formRowNV(title, *args))

        add_text(PROTOCOL_TITLE)
        addForm('Характеристика', 'INF', 'Значення', 'SUP', 'SKV')
        add_text()

        addForm("Сер арифметичне X",
                f"{data.x.x_-data.x.det_x_:.5}",
                f"{data.x.x_:.5}",
                f"{data.x.x_+data.x.det_x_:.5}",
                f"{data.x.det_x_:.5}")
        addForm("Сер арифметичне Y",
                f"{data.y.x_-data.y.det_x_:.5}",
                f"{data.y.x_:.5}",
                f"{data.y.x_+data.y.det_x_:.5}",
                f"{data.y.det_x_:.5}")

        add_text()
        addForm("Коефіціент кореляції",
                f"{data.r_det_v - data.det_r:.5}",
                f"{data.r:.5}",
                f"{data.r_det_v + data.det_r:.5}",
                f"{data.det_r:.5}")

        addForm("Коеф кореляційного відношення p",
                f"{data.det_less_po:.5}",
                f"{data.po_2:.5}",
                f"{data.det_more_po:.5}", "")

        add_text()
        addForm("Ранговий коеф кореляції Спірмена",
                f"{data.teta_c - data.det_teta_c:.5}",
                f"{data.teta_c:.5}",
                f"{data.teta_c + data.det_teta_c:.5}",
                f"{data.det_teta_c:.5}")

        addForm("Ранговий коефіцієнт Кендалла",
                f"{data.teta_k - data.det_teta_k:.5}",
                f"{data.teta_k:.5}",
                f"{data.teta_k + data.det_teta_k:.5}",
                f"{data.det_teta_k:.5}")

        addForm("Коефіцієнт сполучень Пірсона",
                "", f"{data.C_Pearson:.5}", "", "")

        addForm("Міра звʼязку Кендалла",
                f"{data.teta_b - data.det_teta_b:.5}",
                f"{data.teta_b:.5}",
                f"{data.teta_b + data.det_teta_b:.5}",
                f"{data.det_teta_b:.5}")

        add_text()
        addForm("Індекс Фехнера", "",
                f"{data.ind_F:.5}", "")
        addForm("Індекс сполучень", "",
                f"{data.ind_Fi:.5}", "")

        addForm("Коефіцієнт зв’язку Юла Q", "",
                f"{data.ind_Q:.5}", "")
        addForm("Коефіцієнт зв’язку Юла Y", "",
                f"{data.ind_Y:.5}", "")

        add_text()
        N = len(data)
        nu = N - 2
        alpha = 1 - data.trust
        addForm("Значимість коефіцієнта кореліції",
                f"{data.r_signif:.5}",
                "<=",
                f"{func.QuantileTStudent(alpha, nu):.5}")
        addForm("Значимість коефіцієнта p, t-test",
                f"{abs(data.po_signif_t):.5}",
                "<=",
                f"{func.QuantileTStudent(1 - data.trust, nu):.5}")
        try:
            f_po = func.QuantileFisher(1 - data.trust,
                                       data.po_k - 1,
                                       N - data.po_k)
        except TypeError:
            print("Error while calculate quantile fisher")
            f_po = 0.0
        addForm("Значимість коефіцієнта p, f-test",
                f"{data.po_signif_f:.5}",
                "<=",
                f"{f_po:.5}")
        addForm("Значимість коефіцієнта Фі",
                f"{data.ind_F_signif:.5}",
                ">=",
                f"{func.QuantilePearson(1 - data.trust, 1):.5}")
        addForm("Значимість коефіцієнта Юла Q",
                f"{abs(data.ind_Q_signif):.5}",
                "<=",
                f"{func.QuantileNorm(1 - data.trust / 2):.5}")
        addForm("Значимість коефіцієнта Юла Y",
                f"{abs(data.ind_Y_signif):.5}",
                "<=",
                f"{func.QuantileNorm(1 - data.trust / 2):.5}")
        addForm("Значимість коефіцієнта Спірмена",
                f"{abs(data.teta_c_signif):.5}",
                "<=",
                f"{func.QuantileTStudent(1 - data.trust / 2, nu):.5}")
        addForm("Значим рангового коеф Кендалла",
                f"{abs(data.teta_k_signif):.5}",
                "<=",
                f"{func.QuantileNorm(1 - data.trust / 2):.5}")
        addForm("Значимість коефіцієнта Пірсона",
                f"{data.C_Pearson_signif:.5}",
                "<=",
                f"{func.QuantilePearson(1 - data.trust, 10):.5}")
        addForm("Значимість коеф Кендалла",
                f"{abs(data.teta_b_signif):.5}",
                "<=",
                f"{func.QuantileNorm(1 - data.trust / 2):.5}")

        if hasattr(data, "line_a"):
            add_text()
            add_text("Параметри лінійної регресії: a + bx")
            add_text("-" * 16)
            addForm("Параметр a",
                    f"{data.line_a - data.det_line_a:.5}",
                    f"{data.line_a:.5}",
                    f"{data.line_a + data.det_line_a:.5}")
            addForm("Параметр b",
                    f"{data.line_b - data.det_line_b:.5}",
                    f"{data.line_b:.5}",
                    f"{data.line_b + data.det_line_b:.5}")

            add_text()

            f = func.QuantileFisher(1 - data.trust, N - 1, N - 3)
            addForm("Адекватність відтвореної моделі регресії",
                    f"{data.line_f_signif:.5}",
                    "<=",
                    f"{f:.5}")

        if hasattr(data, "parab_a"):
            add_text()
            add_text("Параметри параболічної регресії: a + bx + cx^2")
            add_text("-" * 16)

            addForm("Параметр a",
                    f"{data.parab_a - data.det_parab_a:.5}",
                    f"{data.parab_a:.5}",
                    f"{data.parab_a + data.det_parab_a:.5}")
            addForm("Параметр b",
                    f"{data.parab_b - data.det_parab_b:.5}",
                    f"{data.parab_b:.5}",
                    f"{data.parab_b + data.det_parab_b:.5}")
            addForm("Параметр c",
                    f"{data.parab_c - data.det_parab_c:.5}",
                    f"{data.parab_c:.5}",
                    f"{data.parab_c + data.det_parab_c:.5}")

            add_text()
            t = func.QuantileTStudent(1 - data.trust / 2, N - 3)
            addForm("Значущість параметра a",
                    f"{data.parab_a_t:.5}", "<=",
                    f"{t:.5}")
            addForm("Значущість параметра b",
                    f"{data.parab_b_t:.5}", "<=",
                    f"{t:.5}")
            addForm("Значущість параметра c",
                    f"{data.parab_c_t:.5}", "<=",
                    f"{t:.5}")

        if hasattr(data, "kvaz_a"):
            t = func.QuantileTStudent(1 - data.trust / 2, N - 3)
            add_text()
            add_text("Параметри квазілійнійної регресії: a * exp(bx)")
            add_text("-" * 16)

            addForm("Параметр a",
                    f"{data.kvaz_a - data.det_kvaz_a:.5}",
                    f"{data.kvaz_a:.5}",
                    f"{data.kvaz_a + data.det_kvaz_a:.5}")
            addForm("Параметр b",
                    f"{data.kvaz_b - data.det_kvaz_b:.5}",
                    f"{data.kvaz_b:.5}",
                    f"{data.kvaz_b + data.det_kvaz_b:.5}")

        return "\n".join(inf_protocol)

    def xixi_test_2d(data: DoubleSampleData, hist_data) -> str:
        res = data.xiXiTest(hist_data)
        crits = f"x_2={data.xixitest_x2:.5} <= {data.xixitest_quant:.5}"
        if res:
            return f"Відтворення двовимірного розподілу адекватне: {crits}"
        else:
            return f"Відтворення двовимірного розподілу неадекватне: {crits}"

#
#
#
#
#
#
#

    @staticmethod
    def get_for_sampling_datas(data: SamplingDatas):
        inf_protocol = []
        def add_text(text=""): inf_protocol.append(text)
        def addForm(title, *args): inf_protocol.append(formRowNV(title, *args))

        add_text(PROTOCOL_TITLE)
        addForm('Характеристика', 'INF', 'Значення', 'SUP', 'SKV')
        add_text()

        for i, s in enumerate(data.samples):
            addForm(f"Сер Арифметичне X{i+1}",
                    s.x_ - s.det_x_,
                    s.x_,
                    s.x_ + s.det_x_,
                    s.det_x_)
            addForm(f"Сер квадратичне X{i+1}",
                    s.Sigma - s.det_Sigma,
                    s.Sigma,
                    s.Sigma + s.det_Sigma,
                    s.det_Sigma)

        add_text()
        add_text("Оцінка дисперсійно-коваріаційної матриці DC:")
        n = len(data.samples)
        addForm("", *[f"X{i+1}" for i in range(n)])
        for i in range(n):
            addForm(f"X{i+1}", *[data.DC[i][j] for j in range(n)])

        add_text()
        add_text("Оцінка кореляційної матриці R:")
        addForm("", *[f"X{i+1}" for i in range(n)])
        for i in range(n):
            addForm(f"X{i+1}", *[data.R[i][j] for j in range(n)])

        add_text()
        addForm("Множинні коефіцієнти кореляції")
        addForm("Якщо нерівність виконується, то, приймаємо H0: r=0")
        for i in range(n):
            addForm(f"Множинний коефіцієнт кореляції X{i+1}",
                    "", data.r_multi[i][0])
            addForm("F-тест коефіцієнту",
                    data.r_multi[i][1],
                    "≥",
                    data.r_multi[i][2])

        add_text()
        add_text("Власні вектори")
        addForm("", *[f"F{i+1}" for i in range(n)] + ["Сума"])
        for i in range(n):
            sum_xk = sum([data.DC_eigenvects[i][j] ** 2 for j in range(n)])
            addForm(f"X{i+1}", *([data.DC_eigenvects[i][j] for j in range(n)] +
                                 [sum_xk]))
        add_text()
        addForm("Власні числа", *[data.DC_eigenval[i] for i in range(n)])
        addForm("Частка %", *[data.DC_eigenval_part[i] for i in range(n)])
        addForm("Накопичена", *[data.DC_eigenval_accum[i] for i in range(n)])

        add_text()
        add_text("Факторний аналіз")
        w = data.fact_mat.shape[1]
        addForm("", *[f"F{i+1}" for i in range(w)])
        for i in range(n):
            addForm(f"X{i+1}", *([data.fact_mat[i][j] for j in range(w)]))

        add_text()
        if hasattr(data, "line_A"):
            add_text("Параметри лінійної регресії: Y = AX")
            add_text("-" * 16)
            addForm("Коефіцієнт детермінації", "",
                    data.line_R)
            addForm("Перевірка значущості регресії",
                    data.line_R_f_test,
                    ">",
                    data.line_R_f_quant)
            add_text()
            addForm("Стандартна похибка регресії",
                    data.det_less_line_Sigma,
                    data.line_S_slide,
                    data.det_more_line_Sigma)
            addForm("σ^2 = σˆ^2",
                    data.line_sigma_signif_f_test,
                    "≤",
                    data.line_sigma_signif_f_quant)
            add_text()
            ak_text = ""
            for k, a in enumerate(data.line_A):
                if a > 0:
                    ak_text += f" + {a:.5}x{k+1}"
                else:
                    ak_text += f" - {-a:.5}x{k+1}"
            add_text(f"y = {data.line_A0:.5}" + ak_text)
            add_text()
            addForm(f"Параметр a{0}", "", data.line_A0)
            add_text()
            for k, a in enumerate(data.line_A):
                addForm(f"Параметр a{k+1}",
                        a - data.line_det_A[k],
                        a,
                        a + data.line_det_A[k],
                        data.line_det_A[k])
                addForm(f"T-Тест a{k+1}",
                        data.line_A_t_test[k],
                        "≤",
                        data.line_A_t_quant)
                addForm(f"Стандартизований параметр a{k+1}", '',
                        data.line_stand_A[k])
                add_text()

        if hasattr(data, "line_var_par"):
            add_text("Параметри лінійного різноманіття:")
            add_text("-" * 16)
            ak_text = ""
            for k, a in enumerate(data.line_var_par[:-1]):
                if a > 0:
                    ak_text += f" + {a:.5}x{k+1}"
                else:
                    ak_text += f" - {-a:.5}x{k+1}"
            add_text()
            add_text(f"y = {data.line_var_par[-1]:.5}" + ak_text)
            add_text()
            addForm(f"Параметр a{0}", "", data.line_var_par[-1])
            add_text()
            for k, a in enumerate(data.line_var_par[:-1]):
                addForm(f"Параметр a{k+1}", "", a)
                add_text()

        if hasattr(data, "signif_weighted_sum"):
            add_text()
            add_text("Оцінка якості кластеризації")
            add_text()
            addForm("Сума («зважена») внутрішньокластерних дисперсій",
                    "", data.signif_weighted_sum)
            addForm("Сума попарних внутрішньокластерних",
                    "", data.signif_pair_sum)
            addForm("Загальна внутрішньокластерна дисперсія",
                    "", data.signif_general_dispersion)
            addForm("Відношення функціоналів",
                    "", data.signif_relation_functionals)

        if hasattr(data, "acc_class"):
            add_text()
            add_text("Оцінка якості класифікації")
            add_text()
            addForm("Точність класифікації", "", data.acc_class)
            addForm("PPV", "", data.ppv_class)
            addForm("TPR", "", data.tpr_class)
            addForm("FPR", "", data.fpr_class)
            addForm("FNR", "", data.fnr_class)

        return "\n".join(inf_protocol)

    @staticmethod
    def get_protocol_time_series(data: TimeSeriesData):
        inf_protocol = []
        def add_text(text=""): inf_protocol.append(text)
        def addForm(title, *args): inf_protocol.append(formRowNV(title, *args))

        add_text(PROTOCOL_TITLE)
        addForm('Характеристика', 'INF', 'Значення', 'SUP', 'SKV')
        add_text()

        addForm("Сер арифметичне", "", data.m)

        addForm("Дисперсія", "", data.Dispersion)
        add_text()

        title = "Критерій знаків"
        if data.critetion_sign > data.critetion_sign_signif:
            addForm(title,
                    data.critetion_sign, ">", data.critetion_sign_signif)
            add_text("Процес має тенденцію до зростання")
        elif data.critetion_sign < data.critetion_sign_signif:
            addForm(title,
                    data.critetion_sign, "<", data.critetion_sign_signif)
            add_text("Процес має тенденцію до спадання")
        else:
            addForm(title,
                    abs(data.critetion_sign), "<=", data.critetion_sign_signif)
            add_text("Процес є стаціонарний")
        add_text()

        title = "Критерій Манна"
        if data.critetion_mann > data.critetion_mann_signif:
            addForm(title,
                    data.critetion_mann, ">", data.critetion_mann_signif)
            add_text("Процес має тенденцію до зростання")
        elif data.critetion_mann < data.critetion_mann_signif:
            addForm(title,
                    data.critetion_mann, "<", data.critetion_mann_signif)
            add_text("Процес має тенденцію до спадання")
        else:
            addForm(title,
                    abs(data.critetion_mann), "<=", data.critetion_mann_signif)
            add_text("Процес є стаціонарний")
        add_text()

        title = "Критерій серій"
        if (data.critetion_series_nu > data.critetion_series_nu_signif and
                data.critetion_series_d < data.critetion_series_d_signif):
            addForm(title,
                    data.critetion_series_nu, ">",
                    data.critetion_series_nu_signif)
            addForm("", data.critetion_series_d, "<",
                    data.critetion_series_d_signif)
            add_text("Процес є стаціонарний")
        else:
            addForm(title,
                    data.critetion_series_nu, ">",
                    data.critetion_series_nu_signif)
            addForm("", data.critetion_series_d, "<",
                    data.critetion_series_d_signif)
            add_text("Процес має тенденцію")
        add_text()

        addForm("Критерій «зростаючих» і «спадаючих» серій",
                data.critetion_series_nu, ">",
                data.critetion_series_nu_signif)
        addForm("", data.critetion_series_d, "<",
                data.critetion_series_d_signif)
        if (data.critetion_series_nu > data.critetion_series_nu_signif and
                data.critetion_series_d > data.critetion_series_d_signif):
            add_text("Процес є стаціонарний")
        else:
            add_text("Процес має тенденцію")
        add_text()

        title = "Критерій Аббе"
        if data.critetion_abbe <= data.critetion_abbe_signif:
            addForm(title,
                    data.critetion_abbe, "<=",
                    data.critetion_abbe_signif)
            add_text("Результати спостережень стохастично незалежні")
        else:
            addForm(title,
                    data.critetion_abbe, ">",
                    data.critetion_abbe_signif)
            add_text("Результати спостережень стохастично залежні")

        return "\n".join(inf_protocol)
