import dash
from dash import dcc, html
import plotly.express as px
from parser import BDDParser
from analysis import detect_anomalies


def launch_dashboard(train_json, classes_file):
    parser = BDDParser(train_json, classes_file)
    dist = parser.get_class_distribution()
    bbox_stats = parser.get_bbox_stats()
    anomalies = detect_anomalies(bbox_stats)

    app = dash.Dash(__name__)

    # Class distribution plot
    fig_dist = px.bar(
        x=list(dist.keys()),
        y=list(dist.values()),
        labels={"x": "Class", "y": "Count"},
        title="Class Distribution in BDD100K",
    )

    # Anomaly table
    anomaly_table = html.Table(
        [
            html.Thead(
                html.Tr([
                    html.Th("Class"),
                    html.Th("Avg W"),
                    html.Th("Avg H"),
                    html.Th("Min Area"),
                    html.Th("Max Area"),
                    html.Th("#Boxes")
                ])
            ),
            html.Tbody([
                html.Tr([
                    html.Td(cls),
                    html.Td(f"{stats['avg_width']:.2f}"),
                    html.Td(f"{stats['avg_height']:.2f}"),
                    html.Td(f"{stats['min_area']:.2f}"),
                    html.Td(f"{stats['max_area']:.2f}"),
                    html.Td(stats['num_boxes'])
                ])
                for cls, stats in anomalies.items()
            ])
        ]
    )

    # Layout
    app.layout = html.Div([
        html.H1("BDD100K Object Detection Dashboard"),
        dcc.Graph(figure=fig_dist),
        html.H2("Bounding Box Anomalies"),
        anomaly_table
    ])

    # Print user-friendly URL
    print("\n✅ Dashboard running at: http://127.0.0.1:8050\n")

    # Still bind to all interfaces (works in Docker too)
    app.run(debug=True, host="0.0.0.0", port=8050)


if __name__ == "__main__":
    json_path = r"c:\Users\abhinav.gadge\Documents\KN_Docs_Induction\data_bdd\bdd100k_labels_release\bdd100k\labels\bdd100k_labels_images_val.json"
    classes_path = r"C:\Users\abhinav.gadge\Documents\KN_Docs_Induction\python_inference_client\python_inference_client\data\classes.txt"
    launch_dashboard(train_json=json_path, classes_file=classes_path)
