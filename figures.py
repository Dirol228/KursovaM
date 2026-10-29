import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle, Polygon

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "results", "figures")
os.makedirs(OUT, exist_ok=True)
FS = 9


def box(ax, x, y, w, h, text, fc="#eaf2fb", ec="#1f4e79", fs=FS, lw=1.2,
        rounded=True):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.02,rounding_size=0.12"
                       if rounded else "square,pad=0",
                       fc=fc, ec=ec, lw=lw)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, x1, y1, x2, y2, text=None, fs=FS - 1, color="black"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", lw=1.2, color=color))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, text, ha="center",
                va="bottom", fontsize=fs)


def fig_bpmn():
    fig, ax = plt.subplots(figsize=(7.8, 4.1))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.2); ax.axis("off")
    # початок
    ax.add_patch(Circle((0.5, 4.3), 0.28, fc="#d9f2d9", ec="green", lw=2))
    ax.text(0.5, 3.75, "Надходження\nзавдання\nunif(5; 2) хв", ha="center",
            va="top", fontsize=FS - 1)
    row1 = [(1.5, "Зайняти\n1 стор. ОЗП"),
            (3.3, "Очікування\nCPU1\n(черга)"),
            (5.1, "Трансляція\nCPU1\nunif(5; 1) хв"),
            (6.9, "ОЗП: 1→2 стор.")]
    for x, t in row1:
        box(ax, x, 3.9, 1.5, 0.9, t)
    arrow(ax, 0.78, 4.3, 1.5, 4.3)
    for (x1, _), (x2, _) in zip(row1[:-1], row1[1:]):
        arrow(ax, x1 + 1.5, 4.35, x2, 4.35)
    # перенесення на другий рядок
    ax.annotate("", xy=(8.7, 2.45), xytext=(7.65, 3.9),
                arrowprops=dict(arrowstyle="-|>", lw=1.2))
    row2 = [(7.3, "Очікування\nCPU2\n(черга)"),
            (5.1, "Редагування CPU2\n2·unif(2,5; 0,5) хв"),
            (2.9, "ОЗП: 2→3 стор."),
            (1.0, "Очікування\nCPU3\n(черга)")]
    ys = 1.7

    box(ax, 7.9, ys, 1.5, 0.9, row2[0][1])
    ax.annotate("", xy=(8.65, 2.6), xytext=(8.65, 2.6))
    box(ax, 5.7, ys, 1.8, 0.9, row2[1][1])
    box(ax, 3.9, ys, 1.4, 0.9, row2[2][1])
    box(ax, 2.1, ys, 1.4, 0.9, row2[3][1])
    arrow(ax, 7.9, ys + 0.45, 7.5, ys + 0.45)
    arrow(ax, 5.7, ys + 0.45, 5.3, ys + 0.45)
    arrow(ax, 3.9, ys + 0.45, 3.5, ys + 0.45)
    # ряд 3
    box(ax, 2.1, 0.3, 1.9, 0.9, "Обробка CPU3\n3·unif(1,5; 0,4) хв")
    box(ax, 4.5, 0.3, 1.8, 0.9, "Звільнити\n3 стор. ОЗП")
    arrow(ax, 2.8, ys, 2.8, 1.2)
    arrow(ax, 4.0, 0.75, 4.5, 0.75)
    ax.add_patch(Circle((7.2, 0.75), 0.28, fc="#f8d7d7", ec="red", lw=3))
    arrow(ax, 6.3, 0.75, 6.92, 0.75)
    ax.text(7.7, 0.75, "Завдання\nвиконано", va="center", fontsize=FS - 1)
    # коментар
    ax.text(0.1, 0.9, "M(t) = Σ сторінок усіх завдань\nу системі (у чергах і на\nпроцесорах)",
            fontsize=FS - 1, va="center",
            bbox=dict(boxstyle="round", fc="#fff8dc", ec="gray"))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_bpmn.png"), dpi=200)
    plt.close(fig)



def queue_symbol(ax, x, y, w=0.7, h=0.6, n=7):
    """Черга за табл. 3.1: прямокутник з вертикальними рисками."""
    ax.add_patch(Rectangle((x, y - h / 2), w, h, fc="white", ec="black",
                           lw=1.2))
    for k in range(1, n):
        ax.plot([x + w * k / n] * 2, [y - h / 2, y + h / 2], "k-", lw=0.8)


def device_symbol(ax, x, y, label, r=0.27):
    ax.add_patch(Circle((x, y), r, fc="white", ec="black", lw=1.2))
    ax.text(x, y, label, ha="center", va="center", fontsize=FS,
            style="italic")


def fig_qn():
    """Схема мережі МО в позначеннях таблиці 3.1 методички."""
    fig, ax = plt.subplots(figsize=(7.8, 3.0))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.8); ax.axis("off")
    y = 2.4
    # надходження: стрілка з підписом
    arrow(ax, 0.1, y, 1.3, y)
    ax.text(0.7, y - 0.35, "t=unif(5, 2)", ha="center", va="top",
            fontsize=FS - 1, style="italic")
    xs = [1.3, 4.3, 7.3]
    t_txt = ["t=unif(5, 1)", "t=unif(2.5, 0.5) на стор.",
             "t=unif(1.5, 0.4) на стор."]
    p_txt = ["p = 1 стор.", "p = 2 стор.", "p = 3 стор."]
    for i, x in enumerate(xs):
        queue_symbol(ax, x + 0.1, y)
        ax.plot([x + 0.8, x + 1.25], [y, y], "k-", lw=1.2)     # відрізок
        device_symbol(ax, x + 1.52, y, f"D{i+1}")
        ax.plot([x + 1.79, x + 2.2], [y, y], "k-", lw=1.2)
        ax.text(x + 1.52, y - 0.45, t_txt[i], ha="center", va="top",
                fontsize=FS - 1, style="italic")
        ax.text(x + 1.0, y + 0.55, f"СМО{i+1}", ha="center",
                fontsize=FS, weight="bold")
        ax.text(x + 1.0, y + 0.9, p_txt[i], ha="center",
                fontsize=FS - 1)
    arrow(ax, 3.5, y, 4.3, y)      # дуги
    arrow(ax, 6.5, y, 7.3, y)
    arrow(ax, 9.5, y, 9.95, y)
    ax.text(0.1, 0.55,
            "Атрибут завдання p — кількість сторінок ОЗП (1 → 2 → 3). "
            "Час обслуговування = сума p незалежних значень.\n"
            "Черги необмежені (max = ∞). Займана пам'ять: "
            "M(t) = 1·n₁(t) + 2·n₂(t) + 3·n₃(t), де nᵢ — число завдань "
            "у СМОi.",
            fontsize=FS - 1.5, va="center",
            bbox=dict(boxstyle="round", fc="#fff8dc", ec="gray"))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_qn.png"), dpi=220)
    plt.close(fig)



def uml_class(ax, x, y, w, title, attrs, methods, fs=9):
    """Клас у стилі draw.io: біле тло, чорна рамка, заголовок та розділювачі."""
    lh = 0.27
    head = 0.38
    h = head + lh * len(attrs) + lh * len(methods) + 0.22
    ax.add_patch(Rectangle((x, y), w, h, fc="white", ec="black", lw=1.2))
    ax.text(x + w / 2, y + h - head / 2, title, ha="center", va="center",
            fontsize=fs + 1, weight="bold")
    cy = y + h - head
    ax.plot([x, x + w], [cy, cy], "k-", lw=1.2)
    for a_ in attrs:
        cy -= lh
        ax.text(x + 0.1, cy + lh / 2 - 0.02, a_, va="center", fontsize=fs)
    cy -= 0.1
    ax.plot([x, x + w], [cy, cy], "k-", lw=1.2)
    cy -= 0.02
    for m in methods:
        cy -= lh
        ax.text(x + 0.1, cy + lh / 2 - 0.02, m, va="center", fontsize=fs)
    return h


def fig_uml():
    """UML-діаграма класів: чорно-біла, у стилі draw.io."""
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    fam = next((n for n in ("Arial", "Liberation Sans", "Helvetica")
                if n in names), "DejaVu Sans")
    with plt.rc_context({"font.family": fam}):
        fig, ax = plt.subplots(figsize=(8.6, 5.6))
        ax.set_xlim(0, 11.6)
        ax.set_ylim(0, 7.6)
        ax.axis("off")
        uml_class(ax, 3.8, 0.2, 4.4, "QNSimulation", [
            "- st: List[StationConfig]", "- ia: Gen",
            "- pages_in: List[int]", "- rng: random.Random",
            "- queue: List[deque[Job]]", "- busy: List[int]",
            "- M: int  (зайнято сторінок ОЗП)", "- t: float  (модельний час)",
            "- _heap: календар подій",
            "- int_M, hist, int_q, int_busy ..."],
            ["+ run(T): dict", "+ results(T): dict",
             "- _push(t, kind, data)", "- _advance(t)", "- _enter(i, job)",
             "- _start_service(i, job)", "- _on_arrival()",
             "- _on_depart(i, job)"])
        uml_class(ax, 0.1, 4.7, 3.2, "StationConfig", [
            "+ name: str", "+ time_fn: Gen", "+ servers: int = 1",
            "+ per_page: bool = True"], [])
        uml_class(ax, 0.1, 1.7, 3.2, "Job", [
            "+ id: int", "+ t_arrive: float", "+ pages: int",
            "+ counted: bool"], [])
        uml_class(ax, 8.8, 5.6, 2.7, "build_d4", [],
                  ["+ build_d4(mean_ia,", "    ia_half, servers,",
                   "    seed, warmup,", "    sample_dt)"])
        uml_class(ax, 8.8, 3.4, 2.7, "Генератори Gen", [],
                  ["+ const(d)", "+ exp(d)", "+ unif(d, s)"])
        # StationConfig ◆— QNSimulation (композиція)
        ax.plot([3.3, 3.8], [5.5, 5.5], "k-", lw=1.2)
        ax.add_patch(Polygon([[3.8, 5.5], [3.65, 5.58], [3.5, 5.5],
                              [3.65, 5.42]], fc="black", ec="black"))
        ax.text(3.35, 5.62, "1..*", fontsize=8)
        # Job ◇— QNSimulation (агрегація)
        ax.plot([3.3, 3.8], [2.5, 2.5], "k-", lw=1.2)
        ax.add_patch(Polygon([[3.8, 2.5], [3.65, 2.58], [3.5, 2.5],
                              [3.65, 2.42]], fc="white", ec="black",
                             lw=1.2))
        ax.text(3.35, 2.62, "0..*", fontsize=8)
        # build_d4 ⇢ QNSimulation
        ax.annotate("", xy=(8.2, 5.4), xytext=(8.8, 6.3),
                    arrowprops=dict(arrowstyle="->", lw=1.1, ls="--",
                                    color="black"))
        ax.text(8.0, 6.0, "«create»", fontsize=8, ha="right")
        # QNSimulation ⇢ Gen
        ax.annotate("", xy=(8.8, 4.3), xytext=(8.2, 4.3),
                    arrowprops=dict(arrowstyle="->", lw=1.1, ls="--",
                                    color="black"))
        ax.text(8.5, 4.4, "«use»", fontsize=8, ha="center")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "fig_uml.png"), dpi=220,
                    facecolor="white")
        plt.close(fig)



def fig_flow():
    fig, ax = plt.subplots(figsize=(7.4, 6.9))
    ax.set_xlim(0, 9.5); ax.set_ylim(0, 8.8); ax.axis("off")
    cx = 3.2
    box(ax, cx - 1.4, 8.0, 2.8, 0.6, "Початок: ініціалізація стану,\nпершого надходження, RESET",
        fc="#d9f2d9")
    box(ax, cx - 1.4, 7.0, 2.8, 0.6, "Вибрати найближчу подію\nз календаря (heapq)")
    # ромб t > T
    ax.add_patch(Polygon([[cx, 6.55], [cx + 1.3, 6.0], [cx, 5.45],
                          [cx - 1.3, 6.0]], fc="#fff3cd", ec="black"))
    ax.text(cx, 6.0, "t > T ?", ha="center", va="center", fontsize=FS)
    box(ax, cx - 1.4, 4.45, 2.8, 0.8,
        "Просунути час до t:\nнакопичити ∫M·dt, ∫M²·dt,\n∫n·dt, ∫Q·dt, ∫busy·dt")
    ax.add_patch(Polygon([[cx, 4.2], [cx + 1.4, 3.6], [cx, 3.0],
                          [cx - 1.4, 3.6]], fc="#fff3cd", ec="black"))
    ax.text(cx, 3.6, "тип події", ha="center", va="center", fontsize=FS)
    box(ax, cx - 1.4, 0.3, 2.8, 0.6, "Обчислити результати,\nвивести статистику",
        fc="#f8d7d7")
    arrow(ax, cx, 8.0, cx, 7.6)
    arrow(ax, cx, 7.0, cx, 6.55)
    arrow(ax, cx, 5.45, cx, 5.25)
    ax.text(cx + 0.1, 5.3, "ні", fontsize=FS - 1)
    arrow(ax, cx, 4.45, cx, 4.2)
    # так -> завершення
    ax.plot([cx - 1.3, 0.5, 0.5], [6.0, 6.0, 0.6], "k-", lw=1.2)
    ax.annotate("", xy=(cx - 1.4, 0.6), xytext=(0.5, 0.6),
                arrowprops=dict(arrowstyle="-|>", lw=1.2))
    ax.text(0.7, 6.1, "так", fontsize=FS - 1)
    br = [(6.0, 5.0, "ARRIVAL:\nстворити завдання, M += pages₁,\nзапланувати наступне\nнадходження, _enter(СМО1)"),
          (6.0, 3.6, "DEPART(i):\nзвільнити пристрій; завдання з\nчерги → на обслуговування;\nM += pages(i+1) − pages(i);\n_enter(СМО i+1) або вихід (M −= pages)"),
          (6.0, 2.2, "RESET: обнулити статистику\n(кінець перехідного періоду)\nSAMPLE: записати M(t), ȳ(t)")]
    for x, y, t in br:
        box(ax, x - 0.6, y - 0.5, 3.8, 1.0, t, fs=FS - 1)
    ax.plot([cx + 1.4, 5.4], [3.6, 3.6], "k-")
    ax.annotate("", xy=(5.4, 3.6), xytext=(cx + 1.4, 3.6),
                arrowprops=dict(arrowstyle="-|>", lw=1.2))
    ax.plot([cx + 0.7, 5.4], [3.3, 2.2], "k-", lw=1)
    ax.annotate("", xy=(5.4, 2.2), xytext=(cx + 0.8, 3.25),
                arrowprops=dict(arrowstyle="-|>", lw=1.2))
    ax.plot([cx + 0.7, 5.4], [3.9, 5.0], "k-", lw=1)
    ax.annotate("", xy=(5.4, 5.0), xytext=(cx + 0.8, 3.95),
                arrowprops=dict(arrowstyle="-|>", lw=1.2))
    # повернення
    ax.plot([9.3, 9.3], [2.2, 7.3], "k-", lw=1.2)
    ax.plot([9.0, 9.3], [2.2, 2.2], "k-", lw=1.2)
    ax.plot([9.0, 9.3], [3.6, 3.6], "k-", lw=1.2)
    ax.plot([9.0, 9.3], [5.0, 5.0], "k-", lw=1.2)
    ax.annotate("", xy=(cx + 1.4, 7.3), xytext=(9.3, 7.3),
                arrowprops=dict(arrowstyle="-|>", lw=1.2))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_flow.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig_bpmn(); fig_qn(); fig_uml(); fig_flow()
    print("Схеми збережено в", OUT)
