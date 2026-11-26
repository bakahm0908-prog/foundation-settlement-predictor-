
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import sklearn
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.patches import Rectangle

def build_pdf(report, info, fig):
    from io import BytesIO
    from datetime import datetime, timezone
    from uuid import uuid4
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from xml.sax.saxutils import escape
    buffer = BytesIO()
    navy = colors.HexColor('#102C42')
    teal = colors.HexColor('#087F8C')
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Brand', fontName='Helvetica-Bold', fontSize=23, leading=27, textColor=navy))
    styles.add(ParagraphStyle(name='Section', fontName='Helvetica-Bold', fontSize=11, leading=15, textColor=teal, spaceBefore=9, spaceAfter=5))
    styles.add(ParagraphStyle(name='SmallNote', fontSize=8, leading=11, textColor=colors.HexColor('#475569')))
    def p(text, style='Normal'):
        return Paragraph(text, styles[style])
    def table(rows, widths):
        t = Table(rows, colWidths=widths)
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#F1F5F9')),('ROWBACKGROUNDS',(0,0),(-1,-1),[colors.HexColor('#F1F5F9'),colors.white]),('TEXTCOLOR',(0,0),(-1,-1),navy),('FONTNAME',(0,0),(-1,-1),'Helvetica'),('FONTSIZE',(0,0),(-1,-1),9),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
        return t
    rid = 'GS-' + uuid4().hex[:10].upper()
    date = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    story = [p('GeoSettle AI','Brand'), p('GEOTECHNICAL INTELLIGENCE | AL-MAARIF UNIVERSITY','SmallNote'), Spacer(1,9), p('<b>Foundation Settlement Analysis Report</b>','Heading2'), p(f'{rid} | Generated {date}','SmallNote'), p('01 / Foundation parameters','Section')]
    keys = list(report)[:7]
    story.append(table([[k, f'{float(report[k]):.2f}'] for k in keys],[325,170]))
    story.append(p('02 / Foundation geometry','Section'))
    img = BytesIO()
    fig.savefig(img, format='png', dpi=150, bbox_inches='tight', pad_inches=0.05)
    img.seek(0)
    picture = Image(img)
    scale = min(495/picture.imageWidth, 145/picture.imageHeight)
    picture.drawWidth = picture.imageWidth*scale
    picture.drawHeight = picture.imageHeight*scale
    story.extend([picture,p('Illustrative cross-section; not a construction drawing.','SmallNote'),p('03 / Prediction results','Section')])
    story.append(p(f'<font color="#087F8C" size="19"><b>{report["Predicted settlement (mm)"]:.2f} mm</b></font>','Heading2'))
    story.append(p('Model: '+escape(str(info['model_name']))+' | Baseline prediction','SmallNote'))
    if "Uncertainty prediction (mm)" in report:
        story.append(p(f'Experimental model: {report["Uncertainty prediction (mm)"]:.2f} mm | {report["Target coverage (%)"]:.0f}% target interval: {report["Lower bound (mm)"]:.2f} to {report["Upper bound (mm)"]:.2f} mm','SmallNote'))
        story.append(p(f'Observed internal coverage: {report["Observed coverage (%)"]:.2f}% (29 test records). Separate model: 93 training, 39 calibration. Coverage assumes exchangeability; not a site-specific guarantee.','SmallNote'))
    else:
        story.append(p('Experimental prediction interval unavailable for these inputs.','SmallNote'))
    story.append(p('04 / Baseline internal held-out test performance','Section'))
    story.append(table([['R2','MAE (mm)','RMSE (mm)'],[f'{info["test_r2"]:.4f}',f'{info["test_mae"]:.2f}',f'{info["test_rmse"]:.2f}']],[165,165,165]))
    story.extend([Spacer(1,6),p('Academic research only. Inputs are within individual training ranges; this does not establish site suitability. Large settlements may be underestimated. Independent external validation is not established. Not for design or construction decisions.','SmallNote'),Spacer(1,5),p('Data source: Shahin, M. A. (2003). Use of Artificial Neural Networks for Predicting Settlement of Shallow Foundations on Cohesionless Soils. PhD thesis, University of Adelaide, Table 5.14.','SmallNote')])
    def footer(c, doc):
        c.setStrokeColor(teal); c.line(50,55,545,55)
        c.setFont('Helvetica-Bold',8); c.setFillColor(navy)
        c.drawCentredString(297.5,43,'Developed by Bakr, Ghasaq & Hajar | Al-Maarif University')
        c.setFont('Helvetica',8); c.drawCentredString(297.5,31,'Supervised by Assistant Lecturer Soura Saeed | Civil Engineering')
        c.drawRightString(545,18,str(doc.page))
    SimpleDocTemplate(buffer,pagesize=A4,rightMargin=50,leftMargin=50,topMargin=35,bottomMargin=68,title='GeoSettle AI - Settlement Report').build(story,onFirstPage=footer,onLaterPages=footer)
    return buffer.getvalue()


st.set_page_config(
    page_title="GeoSettle AI | Foundation Research",
    page_icon="G",
    layout="wide"
)

BASE = Path(__file__).resolve().parent

# Load the trained model and research information
@st.cache_resource
def load_model():
    with open(BASE / "model_info.json") as f:
        info = json.load(f)

    if sklearn.__version__ != info["sklearn_version"]:
        raise RuntimeError(
            "Scikit-learn version does not match "
            "the training environment."
        )

    model = joblib.load(
        BASE / "settlement_model.joblib"
    )
    return model, info

try:
    model, info = load_model()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()


@st.cache_resource
def load_uncertainty():
    with open(BASE / "uncertainty_info.json") as f:
        meta = json.load(f)
    if sklearn.__version__ != meta["sklearn_version"]:
        raise RuntimeError("Uncertainty model version mismatch.")
    return joblib.load(BASE / "uncertainty_model.joblib"), meta

try:
    uncertainty_model, uncertainty_info = load_uncertainty()
except Exception:
    uncertainty_model, uncertainty_info = None, None

import plotly.graph_objects as go

st.markdown("""<style>
:root {--navy:#102c42;--teal:#087f8c;}
.stApp {background:#f4f7fa;color:#102c42;}
.block-container {max-width:1280px;padding-top:4rem;padding-bottom:2rem;}
[data-testid="stSidebar"] {background:#eaf0f5;border-right:1px solid #dce5ed;}
[data-testid="stMetric"] {background:white;border:1px solid #dce5ed;border-radius:14px;padding:18px;}
.stButton button,.stDownloadButton button {border-radius:10px;min-height:44px;}
[data-testid="stSidebar"] .stButton button {justify-content:flex-start;text-align:left;}
.hero {display:flex;align-items:center;gap:24px;background:linear-gradient(120deg,#102238,#16465b);color:white;padding:clamp(24px,4vw,48px);border-radius:22px;margin:12px 0 18px;}
.hero-copy {flex:1;min-width:0;}.hero-art {width:38%;max-width:340px;}
.hero h1 {color:white;font-size:clamp(30px,4vw,52px);line-height:1.12;margin:14px 0;letter-spacing:-1px;}
.hero p {color:#d1e2ee;font-size:17px;line-height:1.6;}
.eyebrow {color:#65e1d5;font-size:12px;letter-spacing:2px;font-weight:700;}
.brand {display:flex;align-items:center;gap:12px;margin-bottom:12px;}.brand-mark {background:#102c42;color:#5eead4;border-radius:12px;padding:10px;font-size:24px;}.brand-name {font-size:23px;font-weight:750;color:#102c42;}.brand-sub {font-size:10px;letter-spacing:1px;color:#64748b;}
.footer {background:#102c42;color:#dce5ed;border-radius:16px;text-align:center;padding:28px 12px;font-size:14px;line-height:1.8;}
.tool-tag {color:#087f8c;font-size:11px;font-weight:700;letter-spacing:1.5px;}
@media(max-width:700px){.hero{flex-direction:column;align-items:flex-start}.hero-art{width:65%;align-self:center}.block-container{padding-left:1rem;padding-right:1rem;}}
</style>""", unsafe_allow_html=True)

pages = ["Home", "Settlement Predictor", "3D Foundation Studio", "Sensitivity Analysis", "Model Reliability", "Knowledge Centre", "Research Methodology", "Research Team"]
if "page" not in st.session_state:
    st.session_state.page = "Home"
def navigate(page):
    st.session_state.page = page
page = st.session_state.page
with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-mark">GS</div><div><div class="brand-name">GeoSettle AI</div><div class="brand-sub">GEOTECHNICAL INTELLIGENCE</div></div></div>', unsafe_allow_html=True)
    st.caption("ENGINEERING TOOLS | VERSION 4.0")
    for item in pages:
        st.button(item, key="nav_"+item, type="primary" if item==page else "secondary", use_container_width=True, on_click=navigate, args=(item,))
    st.divider()
    st.caption("Developed by Bakr, Ghasaq & Hajar")
    st.caption("Supervised by Assistant Lecturer Soura Saeed")
    st.caption("Al-Maarif University | Civil Engineering")

st.caption("GEOSETTLE AI / " + page.upper())
st.warning("Academic research prototype for shallow foundations on cohesionless soils. Not for professional design or construction decisions.")

labels = {"B_m":"Foundation width B (m)","q_kPa":"Applied pressure q (kPa)","SPT_N":"Average SPT N","L_over_B":"Length / width ratio (L/B)","Df_over_B":"Embedment ratio (Df/B)"}
defaults = {"B_m":2.0,"q_kPa":150.0,"SPT_N":20.0,"L_over_B":1.0,"Df_over_B":0.5}
if "foundation_values" not in st.session_state:
    st.session_state.foundation_values = defaults.copy()

def save_input(feature):
    st.session_state.foundation_values[feature] = st.session_state["input_"+feature]

if page in ["Settlement Predictor", "3D Foundation Studio", "Sensitivity Analysis"]:
    st.title(page)
    with st.expander("Foundation and soil parameters", expanded=page=="Settlement Predictor"):
        cols=st.columns(2)
        for i, feature in enumerate(labels):
            with cols[i%2]:
                st.number_input(labels[feature], min_value=1.0 if feature=="L_over_B" else (0.0 if feature=="Df_over_B" else 0.01), value=float(st.session_state.foundation_values[feature]), step=10.0 if feature=="q_kPa" else (1.0 if feature=="SPT_N" else 0.1), key="input_"+feature, on_change=save_input, args=(feature,))
    values=st.session_state.foundation_values.copy()
    B,q,N,LB,DfB=[values[k] for k in labels]
    L=B*LB
    Df=B*DfB
    input_data=pd.DataFrame([values],columns=info["features"])
    outside=[]
    for feature,value in values.items():
        limits=info["training_ranges"][feature]
        if not limits["min"] <= value <= limits["max"]:
            outside.append(f'{labels[feature]}: {value:.2f} (training range {limits["min"]:.2f}-{limits["max"]:.2f})')
    if outside:
        st.error("Outside training ranges: " + "; ".join(outside))
    else:
        st.caption("Within individual training ranges; this does not guarantee reliability.")

if page == "Home":
    st.markdown("""<div class="hero"><div class="hero-copy"><div class="eyebrow">ENGINEERING / SCIENCE / INNOVATION</div><h1>Engineering begins beneath the surface.</h1><p>Explore foundation settlement through machine learning, geotechnical research and interactive engineering tools.</p></div><div class="hero-art"><svg viewBox="0 0 260 260" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Illustrative foundation cross section"><path d="M12 130 L135 68 L250 132 L128 197Z" fill="#a99c7c"/><path d="M12 130 L128 197 L128 250 L12 184Z" fill="#77654f"/><path d="M128 197 L250 132 L250 184 L128 250Z" fill="#927d5f"/><path d="M12 146 L128 213 L250 148 M12 162 L128 229 L250 164" fill="none" stroke="#c4ac86"/><path d="M65 132 L134 96 L196 132 L127 169Z" fill="#dce5ea"/><path d="M65 132 L127 169 L127 188 L65 151Z" fill="#8499aa"/><path d="M127 169 L196 132 L196 151 L127 188Z" fill="#a9bbc8"/><path d="M113 57 L137 44 L157 56 L133 69Z" fill="#e2e8f0"/><path d="M113 57 L133 69 L133 145 L113 133Z" fill="#8297aa"/><path d="M133 69 L157 56 L157 132 L133 145Z" fill="#b8c7d3"/><path d="M135 6 V34 M128 26 L135 36 L142 26" fill="none" stroke="#5eead4" stroke-width="3"/><text x="149" y="20" fill="#5eead4" font-size="11">q (kPa)</text></svg></div></div>""", unsafe_allow_html=True)
    st.button("Explore settlement prediction", type="primary", on_click=navigate, args=("Settlement Predictor",))
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Research records",info["total_records"])
    c2.metric("Held-out test R2",f'{info["test_r2"]:.4f}')
    c3.metric("Test MAE",f'{info["test_mae"]:.2f} mm')
    c4.metric("Test RMSE",f'{info["test_rmse"]:.2f} mm')
    st.subheader("Explore our engineering tools")
    tools=[("Settlement Predictor","Estimate settlement using the trained model."),("3D Foundation Studio","Rotate and explore illustrative foundation geometry."),("Sensitivity Analysis","Investigate the effect of changing individual parameters."),("Model Reliability","Review validation results, errors and training ranges."),("Knowledge Centre","Search explanations of geotechnical concepts."),("Research Methodology","Discover the dataset, methods and limitations.")]
    cols=st.columns(2)
    for i,(name,description) in enumerate(tools):
        with cols[i%2]:
            with st.container(border=True):
                st.markdown(f'<div class="tool-tag">ENGINEERING TOOL / {i+1:02d}</div>', unsafe_allow_html=True)
                st.subheader(name)
                st.write(description)
                st.button("Explore tool",key="home_"+name,on_click=navigate,args=(name,),use_container_width=True)
    st.caption("R2 is not percentage accuracy. Testing is internal to the source dataset; external generalizability remains unverified.")

elif page == "Settlement Predictor":
    a,b=st.columns(2)
    with a:
        st.metric("Foundation length",f"{L:.2f} m")
        st.metric("Embedment depth",f"{Df:.2f} m")
        st.write("The model estimates measured final settlement in millimetres.")
        if st.button("Predict Foundation Settlement",type="primary",disabled=bool(outside),use_container_width=True):
            st.session_state.predicted_inputs=values.copy()
    with b:
        st.header("Foundation Geometry")

        fig, ax = plt.subplots(figsize=(7, 5))

        thickness = B * 0.18
        column_width = B * 0.25

        ax.add_patch(Rectangle(
            (-B, -Df - B),
            3 * B,
            Df + B,
            facecolor="#D2B48C",
            alpha=0.55,
            hatch=".."
        ))

        ax.axhline(
            0, color="saddlebrown", linewidth=2
        )

        ax.add_patch(Rectangle(
            (0, -Df),
            B,
            thickness,
            facecolor="gray",
            edgecolor="black"
        ))

        ax.add_patch(Rectangle(
            ((B-column_width)/2, -Df+thickness),
            column_width,
            Df+B*0.5-thickness,
            facecolor="gray",
            edgecolor="black"
        ))

        ax.annotate(
            "",
            xy=(B/2, B*0.52),
            xytext=(B/2, B*0.95),
            arrowprops=dict(
                arrowstyle="->",
                color="red",
                lw=2
            )
        )

        ax.text(
            B/2, B*1.05,
            f"q = {q:.1f} kPa",
            ha="center",
            color="red"
        )

        ax.annotate(
            "",
            xy=(0, -Df-B*0.25),
            xytext=(B, -Df-B*0.25),
            arrowprops=dict(arrowstyle="<->")
        )

        ax.text(
            B/2, -Df-B*0.4,
            f"B = {B:.2f} m",
            ha="center"
        )

        ax.text(
            B*1.15, -Df/2,
            f"Df = {Df:.2f} m"
        )

        ax.set_xlim(-B*0.8, B*2)
        ax.set_ylim(-Df-B*0.8, B*1.3)
        ax.set_aspect("equal")
        ax.axis("off")

        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "Schematic cross-section. "
            "Footing thickness is illustrative."
        )

    if not outside and st.session_state.get("predicted_inputs")==values:
        prediction = float(model.predict(input_data)[0])

        st.header("Prediction Results")

        st.metric(
            "Predicted Settlement",
            f"{prediction:.2f} mm"
        )

        st.write(
            f"Prediction model: {info['model_name']}"
        )

        st.warning(
            "This is a statistical estimate, not a "
            "guaranteed settlement value. The model may "
            "substantially underestimate large settlements."
        )

        report = {
            "Foundation width (m)": B,
            "Foundation length (m)": L,
            "Foundation depth (m)": Df,
            "Applied pressure (kPa)": q,
            "Average SPT N": N,
            "L/B": LB,
            "Df/B": DfB,
            "Predicted settlement (mm)": prediction,
            "Model": info["model_name"],
            "Testing R2": info["test_r2"],
            "Testing MAE (mm)": info["test_mae"],
            "Testing RMSE (mm)": info["test_rmse"],
            "Research team": "Bakr, Ghasaq & Hajar",
            "Institution": "Al-Maarif University",
            "Usage": "Academic research only"
        }

        st.subheader("Scientific prediction uncertainty")
        if uncertainty_model is None:
            st.info("Experimental uncertainty model is not available.")
        elif any(not uncertainty_info["training_ranges"][f]["min"] <= float(input_data[f].iloc[0]) <= uncertainty_info["training_ranges"][f]["max"] for f in uncertainty_info["features"]):
            st.warning("Inputs outside the experimental model training ranges. Interval not reported.")
        else:
            level = st.selectbox("Target coverage", [80, 90, 95], index=1, format_func=lambda x: f"{x}%")
            estimate = float(uncertainty_model.predict(input_data[uncertainty_info["features"]])[0])
            margin = uncertainty_info["uncertainty_margins"][str(level)]
            lo, hi = max(0, estimate-margin), estimate+margin
            result = next(r for r in uncertainty_info["coverage_results"] if r["Target coverage (%)"] == level)
            c1, c2, c3 = st.columns(3)
            c1.metric("Experimental prediction", f"{estimate:.2f} mm")
            c2.metric("Prediction interval", f"{lo:.2f} - {hi:.2f} mm")
            c3.metric("Observed test coverage", f'{result["Observed coverage (%)"]:.2f}%')
            interval_fig = go.Figure(go.Scatter(x=[estimate], y=["Experimental model"], mode="markers", marker=dict(size=12,color="#087f8c"), error_x=dict(type="data", symmetric=False, array=[hi-estimate], arrayminus=[estimate-lo], color="#087f8c", thickness=3)))
            interval_fig.update_layout(height=180, xaxis_title="Settlement (mm)", xaxis=dict(range=[0,max(hi*1.1,1)]), margin=dict(l=10,r=20,t=10,b=40))
            st.plotly_chart(interval_fig, use_container_width=True)
            st.caption(f'{result["Covered records"]} of 29 test records captured. Separate Random Forest trained on 93 records and calibrated on 39. Baseline prediction above is unchanged.')
            st.warning("Exploratory internal analysis. Coverage depends on exchangeability and is not guaranteed for new sites or individual foundations. Not for design or construction decisions.")
            report.update({"Uncertainty prediction (mm)": estimate, "Target coverage (%)": level, "Lower bound (mm)": lo, "Upper bound (mm)": hi, "Observed coverage (%)": result["Observed coverage (%)"]})
        report_df = pd.DataFrame([report])
        try:
            pdf_data = build_pdf(report, info, fig)
            st.download_button("Download Professional Report (PDF)", data=pdf_data,
                               file_name="GeoSettle_Settlement_Report.pdf", mime="application/pdf")
        except ImportError:
            st.info("PDF export requires reportlab in requirements.txt.")

        st.download_button(
            "Download Prediction Report (CSV)",
            data=report_df.to_csv(index=False),
            file_name="settlement_prediction.csv",
            mime="text/csv"
        )


elif page == '3D Foundation Studio':
    st.subheader("Interactive Foundation Visualization")

    def cuboid(x, y, z, dx, dy, dz, color,
               name, opacity=1.0):

        return go.Mesh3d(
            x=[
                x, x+dx, x+dx, x,
                x, x+dx, x+dx, x
            ],
            y=[
                y, y, y+dy, y+dy,
                y, y, y+dy, y+dy
            ],
            z=[
                z, z, z, z,
                z+dz, z+dz, z+dz, z+dz
            ],
            i=[0,0,4,4,0,0,1,1,2,2,3,3],
            j=[1,2,5,6,1,5,2,6,3,7,0,4],
            k=[2,3,6,7,5,4,6,5,7,6,4,7],
            color=color,
            opacity=opacity,
            name=name,
            flatshading=True,
            showlegend=True
        )

    t = B * 0.18
    cw = B * 0.25

    fig3d = go.Figure()

    # Soil mass
    fig3d.add_trace(
        cuboid(
            -B, -B, -Df-B,
            3*B, L+2*B, Df+B,
            "#C4A477", "Soil", 0.25
        )
    )

    # Concrete footing
    fig3d.add_trace(
        cuboid(
            0, 0, -Df,
            B, L, t,
            "#94A3B8", "Concrete footing"
        )
    )

    # Column
    fig3d.add_trace(
        cuboid(
            (B-cw)/2,
            (L-cw)/2,
            -Df+t,
            cw, cw,
            Df+B*0.6-t,
            "#64748B", "Column"
        )
    )

    # Applied loading direction
    fig3d.add_trace(
        go.Cone(
            x=[B/2],
            y=[L/2],
            z=[B*1.1],
            u=[0],
            v=[0],
            w=[-B*0.4],
            anchor="tail",
            colorscale=[
                [0, "#EF4444"],
                [1, "#EF4444"]
            ],
            showscale=False,
            name="Loading direction"
        )
    )

    fig3d.update_layout(
        height=550,
        scene=dict(
            xaxis_title="Width (m)",
            yaxis_title="Length (m)",
            zaxis_title="Elevation (m)",
            aspectmode="data"
        ),
        margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(
            orientation="h"
        )
    )

    st.plotly_chart(
        fig3d,
        use_container_width=True
    )

    a, b, c = st.columns(3)

    a.metric("Width", f"{B:.2f} m")
    b.metric("Length", f"{L:.2f} m")
    c.metric("Embedment", f"{Df:.2f} m")

    st.caption(
        f"Applied pressure: {q:.2f} kPa"
    )

    st.info(
        "Rotate the foundation using your mouse. "
        "Scroll to zoom in and out."
    )

    st.caption(
        "Illustrative geometry only. "
        "Column and footing thicknesses are assumed. "
        "The arrow indicates loading direction, "
        "not a calculated structural force."
    )


elif page == 'Sensitivity Analysis':
    st.subheader("Parametric Sensitivity Analysis")

    st.write(
        "Investigate how changing one input "
        "parameter affects predicted settlement "
        "while holding the other inputs constant."
    )

    labels = {
        "B_m": "Foundation Width (m)",
        "q_kPa": "Applied Pressure (kPa)",
        "SPT_N": "Average SPT N",
        "L_over_B": "Length / Width Ratio",
        "Df_over_B": "Embedment Ratio"
    }

    parameter = st.selectbox(
        "Select parameter to investigate",
        options=list(labels.keys()),
        format_func=lambda x: labels[x]
    )

    points = st.slider(
        "Number of analysis points",
        min_value=20,
        max_value=100,
        value=50,
        step=10
    )

    lower = info["training_ranges"][parameter]["min"]
    upper = info["training_ranges"][parameter]["max"]

    st.write(
        f"Training range: {lower:.2f} - {upper:.2f}"
    )

    if outside:

        st.error(
            "Return all foundation inputs to their "
            "training ranges before running analysis."
        )

    else:

        parameter_values = np.linspace(
            lower, upper, points
        )

        sensitivity_data = pd.DataFrame(
            [values.copy() for _ in range(points)]
        )

        sensitivity_data[parameter] = parameter_values

        settlements = model.predict(
            sensitivity_data[info["features"]]
        )

        sensitivity_results = pd.DataFrame({
            labels[parameter]: parameter_values,
            "Predicted Settlement (mm)": settlements
        })

        fig_s = go.Figure()

        fig_s.add_trace(
            go.Scatter(
                x=parameter_values,
                y=settlements,
                mode="lines+markers",
                name="AI prediction",
                line=dict(
                    color="#38BDF8",
                    width=3
                )
            )
        )

        baseline = float(
            model.predict(input_data)[0]
        )

        fig_s.add_trace(
            go.Scatter(
                x=[values[parameter]],
                y=[baseline],
                mode="markers",
                marker=dict(
                    size=12,
                    color="#EF4444"
                ),
                name="Current foundation"
            )
        )

        fig_s.update_layout(
            title="Influence on Predicted Settlement",
            xaxis_title=labels[parameter],
            yaxis_title="Predicted Settlement (mm)",
            height=450,
            hovermode="closest"
        )

        st.plotly_chart(
            fig_s,
            use_container_width=True
        )

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "Minimum Prediction",
            f"{np.min(settlements):.2f} mm"
        )

        m2.metric(
            "Maximum Prediction",
            f"{np.max(settlements):.2f} mm"
        )

        m3.metric(
            "Prediction Variation",
            f"{np.ptp(settlements):.2f} mm"
        )

        st.dataframe(
            sensitivity_results,
            hide_index=True
        )

        st.download_button(
            "Download Sensitivity Analysis",
            sensitivity_results.to_csv(index=False),
            file_name="sensitivity_analysis.csv",
            mime="text/csv"
        )

        st.warning(
            "These curves represent the trained model's "
            "behavior, not established physical laws. "
            "Some parameter combinations may be poorly "
            "represented in the training dataset, even "
            "when individual inputs are within range. "
            "The prediction variation is not an "
            "uncertainty or confidence interval."
        )


elif page == 'Model Reliability':
    st.header("Model Performance and Reliability")
    if uncertainty_info is not None:
        st.subheader("Experimental conformal prediction")
        comparison = pd.DataFrame([{"Model": "Baseline", "Training records": info["training_records"], "R2": info["test_r2"], "MAE (mm)": info["test_mae"], "RMSE (mm)": info["test_rmse"]}, {"Model": "Uncertainty model", "Training records": uncertainty_info["training_records"], "R2": uncertainty_info["test_r2"], "MAE (mm)": uncertainty_info["test_mae"], "RMSE (mm)": uncertainty_info["test_rmse"]}])
        st.dataframe(comparison.round(4), hide_index=True)
        coverage_table = pd.DataFrame(uncertainty_info["coverage_results"])
        st.dataframe(coverage_table.round(2), hide_index=True)
        st.bar_chart(coverage_table.set_index("Target coverage (%)")[["Observed coverage (%)"]].assign(Target=coverage_table["Target coverage (%)"].to_numpy()))
        st.caption("93 training, 39 calibration, 28 reserved validation and 29 test records. The test split was previously evaluated; these results are exploratory, not independent external validation. The 90% target was not reached on this test split. Wider intervals trade precision for coverage.")
    st.subheader("Baseline model")

    st.write(
        "Selected model:", info["model_name"]
    )

    st.subheader("Experimental Data Division")

    split = pd.DataFrame({
        "Subset": [
            "Training",
            "Validation",
            "Testing"
        ],
        "Records": [
            info["training_records"],
            info["validation_records"],
            info["testing_records"]
        ]
    })

    st.dataframe(split, hide_index=True)

    st.bar_chart(
        split.set_index("Subset")
    )

    st.subheader("Initial Validation Results")

    validation = pd.DataFrame({
        "Model": [
            "Random Forest",
            "Gradient Boosting",
            "Linear Regression",
            "Artificial Neural Network"
        ],
        "MAE (mm)": [
            8.8242, 8.4343, 15.3833, 17.1762
        ],
        "RMSE (mm)": [
            13.8851, 14.0786, 18.2196, 37.9771
        ],
        "R2": [
            0.6859, 0.6771, 0.4592, -1.3496
        ]
    })

    st.dataframe(
        validation,
        hide_index=True,
        use_container_width=True
    )

    st.bar_chart(
        validation.set_index("Model")["RMSE (mm)"]
    )

    st.subheader("Training Parameter Ranges")

    ranges = pd.DataFrame(
        info["training_ranges"]
    ).T

    st.dataframe(ranges)

    st.info(
        "R2 is not percentage prediction accuracy. "
        "MAE and RMSE describe aggregate testing errors, "
        "not uncertainty bounds for individual predictions."
    )


elif page == 'Research Methodology':
    st.header("Research Methodology")

    st.subheader("Graduation Thesis")

    st.write(
        "Reliability and Generalizability of AI-Based "
        "Foundation Settlement Prediction under Limited "
        "Data and Geotechnical Variability."
    )

    st.subheader("Research Database")

    st.write(
        "The dataset contains 189 foundation settlement "
        "records obtained from Shahin (2003), Table 5.14."
    )

    st.subheader("Machine Learning Models")

    st.markdown("""
    - Linear Regression
    - Random Forest
    - Gradient Boosting
    - Artificial Neural Network
    """)

    st.write(
        "Random Forest was selected based on the lowest "
        "validation RMSE among the four initial models."
    )

    st.subheader("Research Limitations")

    st.markdown("""
    - Limited dataset size.
    - Potential relationships between observations
      from the same site.
    - Limited representation of geotechnical variability.
    - No independent external dataset validation.
    - Intended for shallow foundations on cohesionless
      soils represented by the source dataset.
    - Not suitable as a standalone engineering design tool.
    """)

    st.subheader("Reference")

    st.write(
        "Shahin, M. A. (2003). Use of Artificial "
        "Neural Networks for Predicting Settlement "
        "of Shallow Foundations on Cohesionless Soils. "
        "PhD Thesis, University of Adelaide."
    )


elif page == 'Research Team':
    st.header("Research & Development Team")

    st.markdown("""
    ### GeoSettle AI

    Intelligent Foundation Settlement Prediction

    **Developed by:**
    """)

    r1, r2, r3 = st.columns(3)

    with r1:
        st.subheader("Bakr")
        st.caption("Civil Engineering Researcher")

    with r2:
        st.subheader("Ghasaq")
        st.caption("Civil Engineering Researcher")

    with r3:
        st.subheader("Hajar")
        st.caption("Civil Engineering Researcher")

    st.divider()

    st.subheader("Academic Supervisor")
    st.markdown("**Assistant Lecturer Soura Saeed**")
    st.divider()

    st.subheader("Academic Affiliation")

    st.write("Al-Maarif University")
    st.write("Department of Civil Engineering")
    st.write("Graduation Research Project")

    st.subheader("Research Title")

    st.write(
        "Reliability and Generalizability of AI-Based "
        "Foundation Settlement Prediction under "
        "Limited Data and Geotechnical Variability."
    )

    st.subheader("Research Objective")

    st.write(
        "To investigate the reliability, limitations "
        "and generalizability of machine learning "
        "models for shallow foundation settlement "
        "prediction using limited geotechnical data."
    )


elif page == "Knowledge Centre":
    st.title("Knowledge Centre")
    st.write("Explore foundational concepts in settlement and geotechnical engineering.")
    articles = [
        ("Foundation settlement", "Settlement is the downward movement of a foundation due to deformation of the supporting ground. Total and differential settlement are both important in serviceability assessment."),
        ("Immediate settlement", "Immediate settlement arises from soil deformation under loading. It is often important in granular soils, but its magnitude depends on stiffness, stress and geometry."),
        ("Consolidation settlement", "Consolidation involves time-dependent volume reduction as excess pore water pressure dissipates, particularly in saturated fine-grained soils. This platform is not a consolidation model."),
        ("Differential settlement", "Unequal movement between foundation locations can cause distortion and cracking. A single settlement prediction does not determine differential settlement."),
        ("Standard Penetration Test (SPT)", "The SPT N-value is the blow count for the final 300 mm of standard penetration after seating. Corrections and averaging depend on the investigation and method. Match the source dataset definition when using this model."),
        ("Foundation width and embedment", "Width B and length L describe plan geometry; Df is embedment depth to the footing base. L/B and Df/B are dimensionless ratios used by this model."),
        ("Applied foundation pressure", "For a centered vertical load with assumed uniform contact pressure, q equals load divided by contact area. Actual pressure distributions may be nonuniform. Use the pressure definition consistent with the source dataset."),
        ("Random Forest and model reliability", "Random Forest combines predictions from multiple decision trees. Internal test metrics summarize errors on held-out observations, not guaranteed accuracy at a new site. External validation is needed."),
        ("Conventional settlement methods", "Elastic approaches and empirical SPT or CPT correlations are used in geotechnical practice. Method selection depends on soil conditions, loading, geometry and applicable standards. This prototype does not implement these methods.")]
    query=st.text_input("Search settlement topics",placeholder="Try SPT, consolidation, pressure...").strip().casefold()
    matches=[(title,body) for title,body in articles if query in (title+" "+body).casefold()]
    st.caption(f"{len(matches)} topics found")
    for title,body in matches:
        with st.expander(title,expanded=bool(query)):
            st.write(body)
    if not matches: st.info("No matching topics. Try a shorter search term.")
    st.caption("Introductory educational explanations, not design guidance. Research dataset reference: Shahin (2003), University of Adelaide, Table 5.14.")

st.divider()
st.markdown('<div class="footer"><strong>GeoSettle AI</strong><br>Developed by Bakr, Ghasaq &amp; Hajar<br>Supervised by Assistant Lecturer Soura Saeed<br>Department of Civil Engineering | Al-Maarif University<br>Academic Research Project | 2026<br>Research prototype - not for professional design decisions.</div>',unsafe_allow_html=True)
