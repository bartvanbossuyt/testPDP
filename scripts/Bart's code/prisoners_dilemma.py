# -*- coding: utf-8 -*-
"""
Prisoner's Dilemma - Interactive Visualization
An educational Streamlit app demonstrating game theory strategies.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from typing import Callable, Tuple, List
from dataclasses import dataclass
from enum import Enum

# Page configuration
st.set_page_config(
    page_title="Prisoner's Dilemma Simulator",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for smooth animations and better styling
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    .strategy-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .score-display {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
    }
    .cooperate {
        color: #2ecc71;
    }
    .defect {
        color: #e74c3c;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
    }
</style>
""", unsafe_allow_html=True)


class Action(Enum):
    COOPERATE = "C"
    DEFECT = "D"


@dataclass
class PayoffMatrix:
    """Standard Prisoner's Dilemma payoff matrix"""
    # (Player A payoff, Player B payoff)
    both_cooperate: Tuple[int, int] = (3, 3)      # Reward
    both_defect: Tuple[int, int] = (1, 1)         # Punishment
    a_defects: Tuple[int, int] = (5, 0)           # Temptation, Sucker
    b_defects: Tuple[int, int] = (0, 5)           # Sucker, Temptation


def get_payoff(action_a: Action, action_b: Action, matrix: PayoffMatrix = PayoffMatrix()) -> Tuple[int, int]:
    """Calculate payoffs for both players based on their actions."""
    if action_a == Action.COOPERATE and action_b == Action.COOPERATE:
        return matrix.both_cooperate
    elif action_a == Action.DEFECT and action_b == Action.DEFECT:
        return matrix.both_defect
    elif action_a == Action.DEFECT and action_b == Action.COOPERATE:
        return matrix.a_defects
    else:
        return matrix.b_defects


# ============== STRATEGIES ==============

def always_cooperate(my_history: List[Action], opp_history: List[Action]) -> Action:
    """Always cooperate, no matter what."""
    return Action.COOPERATE


def always_defect(my_history: List[Action], opp_history: List[Action]) -> Action:
    """Always defect, no matter what."""
    return Action.DEFECT


def random_strategy(my_history: List[Action], opp_history: List[Action]) -> Action:
    """Randomly cooperate or defect with 50% probability."""
    return Action.COOPERATE if np.random.random() < 0.5 else Action.DEFECT


def tit_for_tat(my_history: List[Action], opp_history: List[Action]) -> Action:
    """Start with cooperation, then copy opponent's last move."""
    if not opp_history:
        return Action.COOPERATE
    return opp_history[-1]


def suspicious_tit_for_tat(my_history: List[Action], opp_history: List[Action]) -> Action:
    """Like Tit-for-Tat, but start with defection."""
    if not opp_history:
        return Action.DEFECT
    return opp_history[-1]


def tit_for_two_tats(my_history: List[Action], opp_history: List[Action]) -> Action:
    """Defect only if opponent defected twice in a row."""
    if len(opp_history) < 2:
        return Action.COOPERATE
    if opp_history[-1] == Action.DEFECT and opp_history[-2] == Action.DEFECT:
        return Action.DEFECT
    return Action.COOPERATE


def friedman_grim_trigger(my_history: List[Action], opp_history: List[Action]) -> Action:
    """Cooperate until opponent defects once, then always defect (unforgiving)."""
    if Action.DEFECT in opp_history:
        return Action.DEFECT
    return Action.COOPERATE


def joss(my_history: List[Action], opp_history: List[Action]) -> Action:
    """Like Tit-for-Tat, but occasionally defect (10% chance) to exploit cooperators."""
    if not opp_history:
        return Action.COOPERATE
    # 10% chance to defect even when opponent cooperated
    if opp_history[-1] == Action.COOPERATE and np.random.random() < 0.1:
        return Action.DEFECT
    return opp_history[-1]


def pavlov(my_history: List[Action], opp_history: List[Action]) -> Action:
    """Win-Stay, Lose-Shift: Repeat last action if it led to good outcome, otherwise switch."""
    if not my_history:
        return Action.COOPERATE
    
    # Check if last round was "good" (3 or 5 points)
    last_my_action = my_history[-1]
    last_opp_action = opp_history[-1]
    my_payoff, _ = get_payoff(last_my_action, last_opp_action)
    
    # Good outcome (R=3 or T=5): stay with same action
    # Bad outcome (P=1 or S=0): switch action
    if my_payoff >= 3:
        return last_my_action
    else:
        return Action.DEFECT if last_my_action == Action.COOPERATE else Action.COOPERATE


def generous_tit_for_tat(my_history: List[Action], opp_history: List[Action]) -> Action:
    """Tit-for-Tat but forgives defection with 10% probability."""
    if not opp_history:
        return Action.COOPERATE
    if opp_history[-1] == Action.DEFECT:
        # 10% chance to forgive
        if np.random.random() < 0.1:
            return Action.COOPERATE
        return Action.DEFECT
    return Action.COOPERATE


def soft_majority(my_history: List[Action], opp_history: List[Action]) -> Action:
    """Cooperate if opponent has cooperated at least half the time."""
    if not opp_history:
        return Action.COOPERATE
    coop_count = sum(1 for a in opp_history if a == Action.COOPERATE)
    return Action.COOPERATE if coop_count >= len(opp_history) / 2 else Action.DEFECT


def hard_majority(my_history: List[Action], opp_history: List[Action]) -> Action:
    """Defect if opponent has defected more than half the time."""
    if not opp_history:
        return Action.DEFECT
    defect_count = sum(1 for a in opp_history if a == Action.DEFECT)
    return Action.DEFECT if defect_count > len(opp_history) / 2 else Action.COOPERATE


def prober(my_history: List[Action], opp_history: List[Action]) -> Action:
    """Start with D, C, C to probe opponent, then decide strategy."""
    round_num = len(my_history)
    if round_num == 0:
        return Action.DEFECT
    elif round_num == 1:
        return Action.COOPERATE
    elif round_num == 2:
        return Action.COOPERATE
    else:
        # If opponent cooperated on rounds 2 and 3, exploit with always defect
        if len(opp_history) >= 3 and opp_history[1] == Action.COOPERATE and opp_history[2] == Action.COOPERATE:
            return Action.DEFECT
        # Otherwise play Tit-for-Tat
        return opp_history[-1]


# Strategy registry with descriptions
STRATEGIES = {
    "Tit-for-Tat": {
        "fn": tit_for_tat,
        "description": "Start by cooperating, then copy opponent's last move. Simple but effective.",
        "author": "Anatol Rapoport",
        "color": "#3498db"
    },
    "Always Cooperate": {
        "fn": always_cooperate,
        "description": "Always cooperate regardless of opponent's actions. Naive but builds trust.",
        "author": "-",
        "color": "#2ecc71"
    },
    "Always Defect": {
        "fn": always_defect,
        "description": "Always defect. Exploits cooperators but scores poorly against itself.",
        "author": "-",
        "color": "#e74c3c"
    },
    "Friedman (Grim Trigger)": {
        "fn": friedman_grim_trigger,
        "description": "Cooperate until opponent defects once, then defect forever. Unforgiving.",
        "author": "James Friedman",
        "color": "#9b59b6"
    },
    "Joss": {
        "fn": joss,
        "description": "Like Tit-for-Tat, but occasionally (10%) defects to exploit cooperators.",
        "author": "Johann Joss",
        "color": "#e67e22"
    },
    "Pavlov": {
        "fn": pavlov,
        "description": "Win-Stay, Lose-Shift: Repeat if outcome was good, switch if bad.",
        "author": "Martin Nowak",
        "color": "#1abc9c"
    },
    "Tit-for-Two-Tats": {
        "fn": tit_for_two_tats,
        "description": "Only defect if opponent defected twice in a row. More forgiving.",
        "author": "-",
        "color": "#34495e"
    },
    "Suspicious Tit-for-Tat": {
        "fn": suspicious_tit_for_tat,
        "description": "Like Tit-for-Tat, but starts with defection.",
        "author": "-",
        "color": "#f39c12"
    },
    "Generous Tit-for-Tat": {
        "fn": generous_tit_for_tat,
        "description": "Tit-for-Tat but forgives defection 10% of the time.",
        "author": "-",
        "color": "#27ae60"
    },
    "Random": {
        "fn": random_strategy,
        "description": "Randomly cooperate or defect with 50% probability.",
        "author": "-",
        "color": "#95a5a6"
    },
    "Soft Majority": {
        "fn": soft_majority,
        "description": "Cooperate if opponent cooperated at least half the time.",
        "author": "-",
        "color": "#16a085"
    },
    "Prober": {
        "fn": prober,
        "description": "Probes opponent with D,C,C, then exploits cooperators or plays Tit-for-Tat.",
        "author": "-",
        "color": "#c0392b"
    },
}


def run_game(strategy_a: str, strategy_b: str, rounds: int, seed: int = None) -> dict:
    """Run a complete game between two strategies."""
    if seed is not None:
        np.random.seed(seed)
    
    fn_a = STRATEGIES[strategy_a]["fn"]
    fn_b = STRATEGIES[strategy_b]["fn"]
    
    history_a: List[Action] = []
    history_b: List[Action] = []
    scores_a: List[int] = []
    scores_b: List[int] = []
    cumulative_a: List[int] = []
    cumulative_b: List[int] = []
    
    total_a, total_b = 0, 0
    
    for round_num in range(rounds):
        action_a = fn_a(history_a, history_b)
        action_b = fn_b(history_b, history_a)
        
        payoff_a, payoff_b = get_payoff(action_a, action_b)
        
        history_a.append(action_a)
        history_b.append(action_b)
        scores_a.append(payoff_a)
        scores_b.append(payoff_b)
        
        total_a += payoff_a
        total_b += payoff_b
        cumulative_a.append(total_a)
        cumulative_b.append(total_b)
    
    return {
        "history_a": history_a,
        "history_b": history_b,
        "scores_a": scores_a,
        "scores_b": scores_b,
        "cumulative_a": cumulative_a,
        "cumulative_b": cumulative_b,
        "total_a": total_a,
        "total_b": total_b
    }


def create_animated_score_chart(results: dict, strategy_a: str, strategy_b: str) -> go.Figure:
    """Create an animated cumulative score chart."""
    rounds = len(results["cumulative_a"])
    
    fig = go.Figure()
    
    # Add traces for both players
    fig.add_trace(go.Scatter(
        x=list(range(1, rounds + 1)),
        y=results["cumulative_a"],
        mode='lines+markers',
        name=f"Speler A: {strategy_a}",
        line=dict(color=STRATEGIES[strategy_a]["color"], width=3),
        marker=dict(size=8),
        hovertemplate="Ronde %{x}<br>Score: %{y}<extra></extra>"
    ))
    
    fig.add_trace(go.Scatter(
        x=list(range(1, rounds + 1)),
        y=results["cumulative_b"],
        mode='lines+markers',
        name=f"Speler B: {strategy_b}",
        line=dict(color=STRATEGIES[strategy_b]["color"], width=3),
        marker=dict(size=8),
        hovertemplate="Ronde %{x}<br>Score: %{y}<extra></extra>"
    ))
    
    # Create animation frames
    frames = []
    for i in range(1, rounds + 1):
        frame = go.Frame(
            data=[
                go.Scatter(
                    x=list(range(1, i + 1)),
                    y=results["cumulative_a"][:i],
                    mode='lines+markers',
                    line=dict(color=STRATEGIES[strategy_a]["color"], width=3),
                    marker=dict(size=8)
                ),
                go.Scatter(
                    x=list(range(1, i + 1)),
                    y=results["cumulative_b"][:i],
                    mode='lines+markers',
                    line=dict(color=STRATEGIES[strategy_b]["color"], width=3),
                    marker=dict(size=8)
                )
            ],
            name=str(i)
        )
        frames.append(frame)
    
    fig.frames = frames
    
    # Add animation controls
    fig.update_layout(
        title=dict(
            text="📈 Cumulatieve Score per Ronde",
            font=dict(size=20)
        ),
        xaxis=dict(
            title="Ronde",
            range=[0, rounds + 1],
            tickmode='linear',
            tick0=0,
            dtick=max(1, rounds // 10)
        ),
        yaxis=dict(
            title="Cumulatieve Score",
            range=[0, max(results["cumulative_a"][-1], results["cumulative_b"][-1]) * 1.1]
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=0,
                x=0.1,
                xanchor="right",
                yanchor="top",
                buttons=[
                    dict(
                        label="▶ Afspelen",
                        method="animate",
                        args=[None, {
                            "frame": {"duration": 100, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 50, "easing": "cubic-in-out"}
                        }]
                    ),
                    dict(
                        label="⏸ Pauze",
                        method="animate",
                        args=[[None], {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0}
                        }]
                    )
                ]
            )
        ],
        sliders=[{
            "active": rounds - 1,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 14},
                "prefix": "Ronde: ",
                "visible": True,
                "xanchor": "center"
            },
            "transition": {"duration": 50, "easing": "cubic-in-out"},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {"args": [[str(i)], {"frame": {"duration": 50, "redraw": True},
                                      "mode": "immediate",
                                      "transition": {"duration": 50}}],
                 "label": str(i),
                 "method": "animate"}
                for i in range(1, rounds + 1)
            ]
        }],
        height=500,
        template="plotly_white"
    )
    
    return fig


def create_action_heatmap(results: dict, strategy_a: str, strategy_b: str) -> go.Figure:
    """Create a heatmap showing actions over time."""
    rounds = len(results["history_a"])
    
    # Convert actions to numeric (0=Cooperate, 1=Defect)
    actions_a = [0 if a == Action.COOPERATE else 1 for a in results["history_a"]]
    actions_b = [0 if a == Action.COOPERATE else 1 for a in results["history_b"]]
    
    fig = go.Figure()
    
    # Create custom colorscale (green=cooperate, red=defect)
    colorscale = [[0, '#2ecc71'], [1, '#e74c3c']]
    
    # Player A actions
    fig.add_trace(go.Heatmap(
        z=[actions_a],
        y=[f"Speler A\n({strategy_a})"],
        x=list(range(1, rounds + 1)),
        colorscale=colorscale,
        showscale=False,
        hovertemplate="Ronde %{x}<br>%{customdata}<extra></extra>",
        customdata=[["Samenwerken" if a == 0 else "Verraden" for a in actions_a]]
    ))
    
    # Player B actions
    fig.add_trace(go.Heatmap(
        z=[actions_b],
        y=[f"Speler B\n({strategy_b})"],
        x=list(range(1, rounds + 1)),
        colorscale=colorscale,
        showscale=False,
        hovertemplate="Ronde %{x}<br>%{customdata}<extra></extra>",
        customdata=[["Samenwerken" if a == 0 else "Verraden" for a in actions_b]]
    ))
    
    fig.update_layout(
        title=dict(
            text="🎯 Acties per Ronde (Groen = Samenwerken, Rood = Verraden)",
            font=dict(size=18)
        ),
        xaxis=dict(title="Ronde", tickmode='linear', tick0=1, dtick=max(1, rounds // 20)),
        yaxis=dict(title=""),
        height=200,
        template="plotly_white"
    )
    
    return fig


def create_payoff_distribution(results: dict, strategy_a: str, strategy_b: str) -> go.Figure:
    """Create a bar chart showing payoff distribution."""
    # Count outcomes
    outcomes = {"R (3,3)": 0, "T (5,0)": 0, "S (0,5)": 0, "P (1,1)": 0}
    
    for a, b in zip(results["history_a"], results["history_b"]):
        if a == Action.COOPERATE and b == Action.COOPERATE:
            outcomes["R (3,3)"] += 1
        elif a == Action.DEFECT and b == Action.DEFECT:
            outcomes["P (1,1)"] += 1
        elif a == Action.DEFECT and b == Action.COOPERATE:
            outcomes["T (5,0)"] += 1
        else:
            outcomes["S (0,5)"] += 1
    
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#95a5a6']
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(outcomes.keys()),
            y=list(outcomes.values()),
            marker_color=colors,
            text=list(outcomes.values()),
            textposition='auto',
            hovertemplate="%{x}<br>Aantal: %{y}<extra></extra>"
        )
    ])
    
    fig.update_layout(
        title=dict(
            text="📊 Verdeling van Uitkomsten",
            font=dict(size=18)
        ),
        xaxis=dict(title="Uitkomst (Speler A, Speler B)"),
        yaxis=dict(title="Aantal rondes"),
        height=350,
        template="plotly_white"
    )
    
    return fig


def run_tournament(strategies: List[str], rounds: int) -> pd.DataFrame:
    """Run a round-robin tournament between all strategies."""
    n = len(strategies)
    scores = {s: 0 for s in strategies}
    matchups = []
    
    for i, strat_a in enumerate(strategies):
        for j, strat_b in enumerate(strategies):
            if i <= j:  # Include self-play
                results = run_game(strat_a, strat_b, rounds)
                scores[strat_a] += results["total_a"]
                if i != j:
                    scores[strat_b] += results["total_b"]
                    matchups.append({
                        "Speler A": strat_a,
                        "Speler B": strat_b,
                        "Score A": results["total_a"],
                        "Score B": results["total_b"]
                    })
                else:
                    matchups.append({
                        "Speler A": strat_a,
                        "Speler B": strat_b,
                        "Score A": results["total_a"],
                        "Score B": results["total_b"]
                    })
    
    return pd.DataFrame(matchups), scores


# ============== MAIN APP ==============

def main():
    st.title("🎲 Prisoner's Dilemma Simulator")
    st.markdown("""
    *Een interactieve verkenning van speltheorie en samenwerking*
    
    Het **Prisoner's Dilemma** is een fundamenteel probleem in speltheorie dat laat zien 
    waarom twee rationele individuen mogelijk niet samenwerken, zelfs als het in hun 
    gezamenlijk belang zou zijn.
    """)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuratie")
        
        mode = st.radio(
            "Modus",
            ["Duel", "Toernooi"],
            help="Duel: twee strategieën tegen elkaar. Toernooi: alle strategieën strijden."
        )
        
        rounds = st.slider(
            "Aantal rondes",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            help="Meer rondes geven strategieën meer tijd om patronen te leren."
        )
        
        st.markdown("---")
        
        st.header("📖 Payoff Matrix")
        st.markdown("""
        |  | Samenwerken | Verraden |
        |---|:---:|:---:|
        | **Samenwerken** | 3, 3 | 0, 5 |
        | **Verraden** | 5, 0 | 1, 1 |
        
        - **R (Reward)**: Beide werken samen → 3 punten elk
        - **T (Temptation)**: Jij verraadt, ander werkt samen → 5 punten
        - **S (Sucker)**: Jij werkt samen, ander verraadt → 0 punten
        - **P (Punishment)**: Beide verraden → 1 punt elk
        """)
    
    if mode == "Duel":
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔵 Speler A")
            strategy_a = st.selectbox(
                "Kies strategie voor Speler A",
                list(STRATEGIES.keys()),
                index=0,
                key="strat_a"
            )
            st.info(STRATEGIES[strategy_a]["description"])
        
        with col2:
            st.subheader("🔴 Speler B")
            strategy_b = st.selectbox(
                "Kies strategie voor Speler B",
                list(STRATEGIES.keys()),
                index=3,  # Friedman
                key="strat_b"
            )
            st.info(STRATEGIES[strategy_b]["description"])
        
        # Run simulation button
        if st.button("🚀 Start Simulatie", type="primary", use_container_width=True):
            with st.spinner("Simulatie wordt uitgevoerd..."):
                results = run_game(strategy_a, strategy_b, rounds)
            
            # Display results
            st.markdown("---")
            st.header("📊 Resultaten")
            
            # Score metrics
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                st.metric(
                    label=f"🔵 {strategy_a}",
                    value=results["total_a"],
                    delta=results["total_a"] - results["total_b"] if results["total_a"] != results["total_b"] else None
                )
            
            with col2:
                winner = "Gelijkspel! 🤝" if results["total_a"] == results["total_b"] else \
                         f"🏆 {strategy_a} wint!" if results["total_a"] > results["total_b"] else \
                         f"🏆 {strategy_b} wint!"
                st.markdown(f"### {winner}")
            
            with col3:
                st.metric(
                    label=f"🔴 {strategy_b}",
                    value=results["total_b"],
                    delta=results["total_b"] - results["total_a"] if results["total_a"] != results["total_b"] else None
                )
            
            # Animated score chart
            st.plotly_chart(
                create_animated_score_chart(results, strategy_a, strategy_b),
                use_container_width=True
            )
            
            # Action heatmap
            st.plotly_chart(
                create_action_heatmap(results, strategy_a, strategy_b),
                use_container_width=True
            )
            
            # Payoff distribution
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(
                    create_payoff_distribution(results, strategy_a, strategy_b),
                    use_container_width=True
                )
            
            with col2:
                # Statistics
                st.subheader("📈 Statistieken")
                
                coop_a = sum(1 for a in results["history_a"] if a == Action.COOPERATE)
                coop_b = sum(1 for a in results["history_b"] if a == Action.COOPERATE)
                
                stats_df = pd.DataFrame({
                    "Statistiek": [
                        "Samenwerking %",
                        "Gemiddelde score/ronde",
                        "Beste ronde",
                        "Slechtste ronde"
                    ],
                    strategy_a: [
                        f"{100*coop_a/rounds:.1f}%",
                        f"{results['total_a']/rounds:.2f}",
                        max(results["scores_a"]),
                        min(results["scores_a"])
                    ],
                    strategy_b: [
                        f"{100*coop_b/rounds:.1f}%",
                        f"{results['total_b']/rounds:.2f}",
                        max(results["scores_b"]),
                        min(results["scores_b"])
                    ]
                })
                st.dataframe(stats_df, hide_index=True, use_container_width=True)
    
    else:  # Tournament mode
        st.subheader("🏆 Toernooi Modus")
        st.markdown("Selecteer welke strategieën meedoen aan het toernooi:")
        
        # Strategy selection
        selected = st.multiselect(
            "Strategieën",
            list(STRATEGIES.keys()),
            default=["Tit-for-Tat", "Always Cooperate", "Always Defect", "Friedman (Grim Trigger)", "Joss", "Pavlov"]
        )
        
        if len(selected) < 2:
            st.warning("Selecteer minstens 2 strategieën voor een toernooi.")
        else:
            if st.button("🚀 Start Toernooi", type="primary", use_container_width=True):
                with st.spinner("Toernooi wordt gespeeld..."):
                    matchups_df, scores = run_tournament(selected, rounds)
                
                st.markdown("---")
                st.header("🏆 Toernooi Resultaten")
                
                # Ranking
                ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                
                # Create ranking chart
                fig = go.Figure(data=[
                    go.Bar(
                        x=[r[1] for r in ranking],
                        y=[r[0] for r in ranking],
                        orientation='h',
                        marker_color=[STRATEGIES[r[0]]["color"] for r in ranking],
                        text=[r[1] for r in ranking],
                        textposition='auto'
                    )
                ])
                
                fig.update_layout(
                    title="Totale Score per Strategie",
                    xaxis_title="Totale Score",
                    yaxis=dict(autorange="reversed"),
                    height=400,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Matchup details
                st.subheader("📋 Wedstrijd Details")
                st.dataframe(matchups_df, hide_index=True, use_container_width=True)
    
    # Educational section
    with st.expander("📚 Leer meer over het Prisoner's Dilemma"):
        st.markdown("""
        ### Achtergrond
        
        Het Prisoner's Dilemma werd in 1950 geformuleerd door Merrill Flood en Melvin Dresher 
        en later verfijnd door Albert Tucker. Het illustreert een situatie waarin individuele 
        rationaliteit leidt tot een collectief suboptimaal resultaat.
        
        ### Het Verhaal
        
        Twee verdachten worden apart ondervraagd. Elk heeft twee opties:
        - **Samenwerken** (zwijgen): Bescherm je partner
        - **Verraden** (bekennen): Verraad je partner
        
        ### Waarom is dit belangrijk?
        
        Het Prisoner's Dilemma modelleert veel echte situaties:
        - Wapenwedloop tussen landen
        - Klimaatonderhandelingen
        - Bedrijfscompetitie
        - Dagelijkse sociale interacties
        
        ### Robert Axelrod's Toernooi (1980)
        
        Robert Axelrod organiseerde een computertoernooi waarbij strategieën tegen elkaar 
        speelden. Verrassend genoeg won de simpelste strategie: **Tit-for-Tat**.
        
        Succesvolle strategieën blijken deze eigenschappen te hebben:
        1. **Aardig**: Begin met samenwerken
        2. **Vergeldend**: Straf verraad
        3. **Vergevingsgezind**: Geef nieuwe kansen
        4. **Duidelijk**: Wees voorspelbaar
        """)


if __name__ == "__main__":
    main()
