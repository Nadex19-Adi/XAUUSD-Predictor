import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_flowchart():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    
    # Title
    ax.text(5, 5.6, "XAUUSD Predictor: System Architecture & Data Flow", 
            fontsize=14, fontweight='bold', ha='center', color='#1a365d')
    
    # Nodes
    boxes = [
        {"text": "Data Ingestion\n(Kaggle CSVs + yfinance)\nResampled to 5m UTC", "x": 0.5, "y": 4.0, "w": 2.2, "h": 1.0, "color": "#ebf8ff", "edge": "#3182ce"},
        {"text": "Feature Engineering\n(14 Indicators: RSI,\nMACD, BB, ATR...)", "x": 3.8, "y": 4.0, "w": 2.2, "h": 1.0, "color": "#ebf8ff", "edge": "#3182ce"},
        {"text": "Sharded RAG Engine\n(ChromaDB)\n[legacy, mid, recent]", "x": 3.8, "y": 1.8, "w": 2.2, "h": 1.0, "color": "#e6fffa", "edge": "#319795"},
        {"text": "XGBoost Classifier\nHist-Based GBDT\n(xgb_model.json)", "x": 7.3, "y": 4.0, "w": 2.2, "h": 1.0, "color": "#feebc8", "edge": "#dd6b20"},
        {"text": "FastAPI Services\nStreamlit Dashboard\n(Live Signal GUI)", "x": 7.3, "y": 1.8, "w": 2.2, "h": 1.0, "color": "#edf2f7", "edge": "#4a5568"},
    ]
    
    for box in boxes:
        rect = patches.FancyBboxPatch((box["x"], box["y"]), box["w"], box["h"], 
                                    boxstyle="round,pad=0.1", 
                                    facecolor=box["color"], edgecolor=box["edge"], linewidth=2)
        ax.add_patch(rect)
        ax.text(box["x"] + box["w"]/2.0, box["y"] + box["h"]/2.0, box["text"], 
                fontsize=9, ha='center', va='center', fontweight='semibold', color='#2d3748')
        
    # Arrows (Lines with annotating markers)
    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#4a5568", lw=1.5, ls="-", shrinkA=2, shrinkB=2))
        if label:
            ax.text((x1+x2)/2.0, (y1+y2)/2.0 + 0.1, label, fontsize=8, ha='center', color='#718096')

    # Data Flow Connectors
    draw_arrow(2.8, 4.5, 3.7, 4.5, "Raw OHLC")
    draw_arrow(6.1, 4.5, 7.2, 4.5, "Features")
    draw_arrow(4.9, 3.9, 4.9, 2.9, "Index Features")
    draw_arrow(6.1, 2.3, 7.2, 4.2, "Similarity Score")
    draw_arrow(8.4, 3.9, 8.4, 2.9, "Signals")
    
    plt.tight_layout()
    plt.savefig("flowchart.png", bbox_inches='tight', dpi=150)
    plt.close()
    print("Generated flowchart.png")

def generate_usecase():
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.axis('off')
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 6)
    
    # Title
    ax.text(4, 5.6, "XAUUSD Predictor: Use Case Diagram", 
            fontsize=14, fontweight='bold', ha='center', color='#1a365d')
    
    # System boundary box
    rect_sys = patches.FancyBboxPatch((2.0, 0.4), 4.0, 4.8, boxstyle="round,pad=0.1", 
                                      facecolor="#f7fafc", edgecolor="#cbd5e0", linewidth=1.5, ls="--")
    ax.add_patch(rect_sys)
    ax.text(4.0, 5.0, "Predictor System Boundary", fontsize=9, fontstyle='italic', ha='center', color='#718096')
    
    # Actors
    ax.text(1.0, 3.0, "Quantitative\nTrader\n(Aditya / Student)", fontsize=10, ha='center', fontweight='bold', color='#2b6cb0')
    ax.text(7.0, 3.0, "External Data\nBroker\n(yfinance API)", fontsize=10, ha='center', fontweight='bold', color='#c53030')
    
    # Use cases inside boundary
    ucs = [
        {"text": "Ingest Market Data", "x": 3.0, "y": 4.0},
        {"text": "Train Predictive Model", "x": 3.0, "y": 3.1},
        {"text": "Query Real-Time Signals", "x": 3.0, "y": 2.2},
        {"text": "Retrieve Past Regimes (RAG)", "x": 3.0, "y": 1.3},
        {"text": "Monitor System Health", "x": 3.0, "y": 0.5},
    ]
    
    for uc in ucs:
        rect = patches.Ellipse((uc["x"] + 1.0, uc["y"] + 0.25), 3.0, 0.6, 
                               facecolor="#ebf8ff", edgecolor="#3182ce", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(uc["x"] + 1.0, uc["y"] + 0.25, uc["text"], 
                fontsize=9, ha='center', va='center', fontweight='semibold', color='#2d3748')
        
    # Draw connections
    # Trader to Use cases
    ax.plot([1.5, 3.0], [3.2, 4.25], color="#4a5568", lw=1.2, ls="-")
    ax.plot([1.5, 3.0], [3.1, 3.35], color="#4a5568", lw=1.2, ls="-")
    ax.plot([1.5, 3.0], [3.0, 2.45], color="#4a5568", lw=1.2, ls="-")
    ax.plot([1.5, 3.0], [2.9, 1.55], color="#4a5568", lw=1.2, ls="-")
    ax.plot([1.5, 3.0], [2.8, 0.75], color="#4a5568", lw=1.2, ls="-")
    
    # Data source to Use case
    ax.plot([6.5, 5.0], [3.0, 4.25], color="#4a5568", lw=1.2, ls="-")
    
    plt.tight_layout()
    plt.savefig("usecase.png", bbox_inches='tight', dpi=150)
    plt.close()
    print("Generated usecase.png")

def generate_mindmap():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    
    # Title
    ax.text(5, 5.6, "XAUUSD RAG Predictor: Concept Mind Map", 
            fontsize=14, fontweight='bold', ha='center', color='#1a365d')
    
    # Central Idea
    rect_ctr = patches.FancyBboxPatch((4.0, 2.5), 2.0, 0.8, boxstyle="round,pad=0.1", 
                                      facecolor="#fed7d7", edgecolor="#e53e3e", linewidth=2.5)
    ax.add_patch(rect_ctr)
    ax.text(5.0, 2.9, "XAUUSD Predictor\nEcosystem", fontsize=11, fontweight='bold', ha='center', va='center', color='#9b2c2c')
    
    # Branches
    branches = [
        {"title": "Data Pipeline", "x": 1.5, "y": 4.5, "w": 2.0, "h": 0.6, "items": ["1.7M Raw Rows", "Resample to 5m", "14 Indicators"], "color": "#ebf8ff", "edge": "#3182ce", "c_color": "#2b6cb0"},
        {"title": "Sharded RAG Engine", "x": 6.5, "y": 4.5, "w": 2.0, "h": 0.6, "items": ["ChromaDB DB", "all-MiniLM-L6-v2", "3-Shard Arch"], "color": "#e6fffa", "edge": "#319795", "c_color": "#234e52"},
        {"title": "XGBoost Classifier", "x": 1.5, "y": 0.8, "w": 2.0, "h": 0.6, "items": ["Hist GBDT", "5-fold Walk-Forward", "xgb_model.json"], "color": "#feebc8", "edge": "#dd6b20", "c_color": "#dd6b20"},
        {"title": "Interface Layer", "x": 6.5, "y": 0.8, "w": 2.0, "h": 0.6, "items": ["FastAPI Backend", "Streamlit Dashboard", "Uptime & Health"], "color": "#edf2f7", "edge": "#4a5568", "c_color": "#2d3748"},
    ]
    
    for br in branches:
        # Branch main node
        rect = patches.FancyBboxPatch((br["x"], br["y"]), br["w"], br["h"], boxstyle="round,pad=0.1", 
                                      facecolor=br["color"], edgecolor=br["edge"], linewidth=2)
        ax.add_patch(rect)
        ax.text(br["x"] + br["w"]/2.0, br["y"] + br["h"]/2.0, br["title"], 
                fontsize=9, fontweight='bold', ha='center', va='center', color=br["c_color"])
        
        # Sub-items
        for idx, item in enumerate(br["items"]):
            offset = 0.4 if br["y"] > 2.5 else -0.4
            item_y = br["y"] + (idx + 1) * offset
            ax.text(br["x"] + br["w"]/2.0, item_y, f"• {item}", fontsize=8, ha='center', color='#4a5568')
            # Connecting line to item
            ax.plot([br["x"] + br["w"]/2.0, br["x"] + br["w"]/2.0], [br["y"] + (0.6 if offset > 0 else 0), item_y], color=br["edge"], lw=0.8, ls=":")
            
    # Central connections
    ax.plot([5.0, 2.5], [3.3, 4.5], color="#a0aec0", lw=1.5)
    ax.plot([5.0, 7.5], [3.3, 4.5], color="#a0aec0", lw=1.5)
    ax.plot([5.0, 2.5], [2.5, 1.4], color="#a0aec0", lw=1.5)
    ax.plot([5.0, 7.5], [2.5, 1.4], color="#a0aec0", lw=1.5)
    
    plt.tight_layout()
    plt.savefig("mindmap.png", bbox_inches='tight', dpi=150)
    plt.close()
    print("Generated mindmap.png")

if __name__ == "__main__":
    generate_flowchart()
    generate_usecase()
    generate_mindmap()
