import os
import cv2
from pathlib import Path
import dash
from dash import dcc, html, dash_table, Input, Output, State
import plotly.express as px
from parser import BDDParser
from analysis import detect_anomalies, find_interesting_samples

# --- Assets folder (Dash automatically serves /assets/)
ASSETS_DIR = "/app/assets"
os.makedirs(ASSETS_DIR, exist_ok=True)

def draw_bboxes(image_path, bboxes, save_name):
    """Draw multiple bounding boxes on an image and save to assets folder."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    for cls_name, bbox in bboxes:
        x, y, w, h = bbox
        x, y, w, h = int(x), int(y), int(w), int(h)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(img, cls_name, (x, max(15, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    save_path = os.path.join(ASSETS_DIR, save_name)
    cv2.imwrite(save_path, img)
    return save_path

def launch_dashboard(train_json, classes_file, images_dir):
    parser = BDDParser(train_json, classes_file)
    dist = parser.get_class_distribution()
    bbox_stats = parser.get_bbox_stats()
    anomalies = detect_anomalies(bbox_stats)
    interesting = find_interesting_samples(parser.annotations, parser.classes)

    app = dash.Dash(__name__)

    # --- Distribution Plot
    fig_dist = px.bar(
        x=list(dist.keys()),
        y=list(dist.values()),
        labels={"x": "Class", "y": "Count"},
        title="Class Distribution in BDD100K",
    )

    # --- Anomalies Table
    anomaly_table = dash_table.DataTable(
        columns=[
            {"name": "Class", "id": "Class"},
            {"name": "Avg W", "id": "Avg W"},
            {"name": "Avg H", "id": "Avg H"},
            {"name": "Min Area", "id": "Min Area"},
            {"name": "Max Area", "id": "Max Area"},
            {"name": "#Boxes", "id": "#Boxes"},
        ],
        data=[
            {
                "Class": cls,
                "Avg W": f"{stats['avg_width']:.2f}",
                "Avg H": f"{stats['avg_height']:.2f}",
                "Min Area": f"{stats['min_area']:.2f}",
                "Max Area": f"{stats['max_area']:.2f}",
                "#Boxes": stats['num_boxes'],
            }
            for cls, stats in anomalies.items()
        ],
        style_table={'overflowX': 'auto'},
    )

    # --- Interesting Samples Table
    sample_rows = []
    for category, samples in interesting.items():
        for s in samples:
            sample_rows.append({
                "Category": category,
                "Image": s["image"],
                "Class": s["class"],
                "Area": f"{s['area']:.1f}",
                "BBox": str(s["bbox"]),
            })

    samples_table = dash_table.DataTable(
        id="samples-table",
        columns=[
            {"name": "Category", "id": "Category"},
            {"name": "Image", "id": "Image"},
            {"name": "Class", "id": "Class"},
            {"name": "Area", "id": "Area"},
            {"name": "BBox", "id": "BBox"},
        ],
        data=sample_rows,
        row_selectable="single",
        style_table={'overflowX': 'auto'},
    )

    app.layout = html.Div([
        html.H1("BDD100K Object Detection Dashboard"),
        dcc.Graph(figure=fig_dist),
        html.H2("Bounding Box Anomalies"),
        anomaly_table,
        html.H2("Interesting / Unique Samples"),
        samples_table,
        html.Div(id="image-preview")
    ])

    # --- Callback: Show selected sample image with all relevant bboxes
    @app.callback(
        Output("image-preview", "children"),
        Input("samples-table", "selected_rows"),
        State("samples-table", "data"),
    )
    def update_preview(selected_rows, table_data):
        if not selected_rows:
            return html.Div("Select a row above to preview image", style={"marginTop": "20px"})

        row = table_data[selected_rows[0]]
        img_name = row["Image"]

        # Gather all bboxes for this image from interesting samples
        bboxes = []
        for r in table_data:
            if r["Image"] == img_name:
                bbox = eval(r["BBox"])
                bboxes.append((r["Class"], bbox))

        # Check possible directories
        img_path = os.path.join(images_dir, img_name)
        if not os.path.exists(img_path):
            return html.Div(f"Image {img_name} not found!", style={"color": "red"})

        save_name = f"{Path(img_name).stem}_preview.jpg"
        draw_bboxes(img_path, bboxes, save_name)

        return html.Div([
            html.H3(f"Preview: {img_name}"),
            html.Img(src=f"/assets/{save_name}", style={"maxWidth": "80%", "marginTop": "10px"})
        ])

    print("\n✅ Dashboard running at: http://127.0.0.1:8050\n")
    app.run(debug=True, host="0.0.0.0", port=8050)

if __name__ == "__main__":
    json_path = "/app/data_bdd/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json"
    classes_path = "/app/data_bdd/classes.txt"
    images_dir = "/app/data_bdd/bdd100k_images_100k/bdd100k/images/100k/train/"  # correct path inside container
    launch_dashboard(train_json=json_path, classes_file=classes_path, images_dir=images_dir)
