from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib
matplotlib.use("Agg")
import os
import datetime
import csv 
import seaborn as sns
import matplotlib.pyplot as plt

# ======================
# INISIALISASI FLASK
# ======================
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "skripsi_smk")


# ======================
# GLOBAL VARIABLE
# ======================
model = None
X_test = y_test = None
import joblib
if os.path.exists("model_rf.pkl"):
    model = joblib.load("model_rf.pkl")

# ======================
# MAPPING KATEGORIK
# ======================
mapping_sikap = {
    "Sangat Baik": 4,
    "Baik": 3,
    "Cukup": 2,
    "Kurang": 1
}

mapping_admin = {
    "Belum": 0,
    "Lunas": 1
}

mapping_kelulusan = {
    "Lulus": 1,
    "Tidak Lulus": 0
}

# ======================
# LOGIN
# ======================
@app.route("/", methods=["GET", "POST"])
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        users = pd.read_csv("users.csv", dtype=str)

        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        role_input = request.form.get("role", "").strip()

        # Cari user berdasarkan username
        user = users[users["username"] == username]

        if user.empty:
            flash("Username tidak ditemukan", "danger")
            return redirect(url_for("login"))

        # Cek password
        if user.iloc[0]["password"] != password:
            flash("Password salah", "danger")
            return redirect(url_for("login"))

        # Cek role
        if user.iloc[0]["role"] != role_input:
            flash("Role yang dipilih tidak sesuai dengan akun", "danger")
            return redirect(url_for("login"))

        # Jika semua cocok, login berhasil
        session.clear()
        session["user"] = username
        session["role"] = role_input
        session["dataset_uploaded"] = False
        session["model_trained"] = False

        if role_input == "admin":
            return redirect(url_for("dashboard"))          # admin
        else:
            return redirect(url_for("dashboard_user"))     # user


    return render_template("login.html")

# ======================
# DASHBOARD USER
# ======================
@app.route("/dashboard-user")
def dashboard_user():
    if "user" not in session or session["role"] != "user":
        return redirect(url_for("login"))

    # Data apa pun yang ingin ditampilkan user, misal status upload dataset, hasil prediksi, dll.
    data = []
    if os.path.exists("riwayat_prediksi.csv"):
        df = pd.read_csv("riwayat_prediksi.csv")
        data = df.to_dict(orient="records")  # semua hasil bisa dilihat user

    return render_template(
        "dashboard_user.html",
        user=session["user"],
        role=session["role"],
        data=data
    )

# ======================
# DASHBOARD ADMIN
# ======================
@app.route("/dashboard")
def dashboard():
    if "user" not in session or session.get("role") != "admin":
        return redirect(url_for("login"))

    return render_template(
        "dashboard.html",
        user=session["user"],
        role=session["role"],              # ⬅️ INI PENTING
        dataset_uploaded=session.get("dataset_uploaded"),
        model_trained=session.get("model_trained")
    )


# ======================
# ABOUT
# ======================
@app.route("/about")
def about():
    if "user" not in session:
        return redirect(url_for("login"))

    return render_template(
        "about.html",
        user=session["user"],
        role=session["role"]
    )

# ======================
# UPLOAD DATASET (ADMIN)
# ======================
@app.route("/upload-dataset", methods=["POST"])
def upload_dataset():
    if "user" not in session or session.get("role") != "admin":
        return redirect(url_for("login"))

    file = request.files.get("dataset")
    if not file or file.filename == "":
        flash("File dataset belum dipilih", "warning")
        return redirect(url_for("dashboard"))

    if not file.filename.lower().endswith(".csv"):
        flash("Format file harus CSV", "danger")
        return redirect(url_for("dashboard"))

    try:
        path = "dataset_upload.csv"
        file.save(path)

        global model, X_train, X_test, y_train, y_test, oob_score

        # ======================
        # LOAD DATASET
        # ======================
        df = pd.read_csv(path)

        # ======================
        # DATA CLEANING
        # ======================

        # 1. Hapus data duplikat
        df = df.drop_duplicates()

        # 2. Tangani missing value
        df = df.dropna()

        # 3. Pastikan kolom numerik valid
        kolom_numerik = [
            "RATA-RATA",
            "NILAI UKK",
            "PRAKERIN",
            "Persentase_Kehadiran"
        ]

        for col in kolom_numerik:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Hapus data yang gagal dikonversi
        df = df.dropna()

        # 4. Validasi rentang nilai
        df = df[
            (df["Persentase_Kehadiran"] >= 0) &
            (df["Persentase_Kehadiran"] <= 100)
        ]

        # ======================
        # ENCODING DATA KATEGORIK
        # ======================
        df["Sikap"] = df["Sikap"].map(mapping_sikap)
        df["Status_Pembayaran_Administrasi"] = df[
            "Status_Pembayaran_Administrasi"
        ].map(mapping_admin)
        df["Status_Kelulusan"] = df["Status_Kelulusan"].map(mapping_kelulusan)

        # Hapus jika ada hasil mapping gagal
        df = df.dropna()

        # ======================
        # PEMISAHAN FITUR & LABEL
        # ======================
        X = df[
            [
                "RATA-RATA",
                "NILAI UKK",
                "PRAKERIN",
                "Persentase_Kehadiran",
                "Sikap",
                "Status_Pembayaran_Administrasi",
            ]
        ]
        y = df["Status_Kelulusan"]

        # ======================
        # SPLIT DATA
        # ======================
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        # ======================
        # TRAINING MODEL
        # ======================
        model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        bootstrap=True,
        oob_score=True,
        random_state=42,
        class_weight="balanced"
        )
        model.fit(X_train, y_train)
        oob_score = model.oob_score_

        # ======================
        # SIMPAN MODEL
        # ======================
        import joblib
        joblib.dump(model, "model_rf.pkl")


        # ======================
        # SESSION STATUS
        # ======================
        session["dataset_uploaded"] = True
        session["model_trained"] = True

        flash("Dataset berhasil diproses, dibersihkan, dan model dilatih", "success")
        return redirect(url_for("dashboard"))

    except Exception as e:
        print("ERROR:", e)
        flash("Dataset tidak sesuai format atau terdapat kesalahan dalam proses", "danger")
        return redirect(url_for("dashboard"))

# ======================
# ANALISIS MODEL (ADMIN)
# ======================
@app.route("/analisis")
def analisis():
    global model, X_train, X_test, y_train, y_test, oob_score
    if "user" not in session or session["role"] != "admin":
        return redirect(url_for("login"))

    if not session.get("model_trained"):
        flash("Upload dan proses dataset terlebih dahulu", "warning")
        return redirect(url_for("dashboard"))

    # =========================
    # LOAD DATASET AWAL
    # =========================
    df_awal = pd.read_csv("dataset_upload.csv")
    jumlah_awal = df_awal.shape[0]
    contoh_data_awal = df_awal.head(5).to_dict(orient="records")

    # =========================
    # REPLIKASI PREPROCESSING
    # (HARUS SAMA DENGAN TRAINING)
    # =========================
    df = df_awal.drop_duplicates()
    df = df.dropna()

    kolom_numerik = [
        "RATA-RATA",
        "NILAI UKK",
        "PRAKERIN",
        "Persentase_Kehadiran"
    ]

    for col in kolom_numerik:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()

    df = df[
        (df["Persentase_Kehadiran"] >= 0) &
        (df["Persentase_Kehadiran"] <= 100)
    ]

    # =========================
    # ENCODING
    # =========================
    df["Sikap"] = df["Sikap"].map(mapping_sikap)
    df["Status_Pembayaran_Administrasi"] = df[
        "Status_Pembayaran_Administrasi"
    ].map(mapping_admin)
    df["Status_Kelulusan"] = df["Status_Kelulusan"].map(mapping_kelulusan)

    df = df.dropna()
    jumlah_bersih = df.shape[0]

    # Contoh data setelah preprocessing
    contoh_data = df.head(5).to_dict(orient="records")

    # =========================
    # SELEKSI FITUR & TARGET
    # =========================
    X = df[
        [
            "RATA-RATA",
            "NILAI UKK",
            "PRAKERIN",
            "Persentase_Kehadiran",
            "Sikap",
            "Status_Pembayaran_Administrasi",
        ]
    ]

    y = df["Status_Kelulusan"]

    # Contoh data fitur & target
    contoh_X = X.head(5).to_dict(orient="records")
    contoh_y = y.head(5).tolist()

    jumlah_fitur = X.shape[1]
    jumlah_data = X.shape[0]

    # =========================
    # INFORMASI DATA SPLIT
    # =========================
    jumlah_train = len(X_train)
    jumlah_test = len(X_test)

    persen_train = round((jumlah_train / (jumlah_train + jumlah_test)) * 100, 2)
    persen_test = round((jumlah_test / (jumlah_train + jumlah_test)) * 100, 2)

    distribusi_train = y_train.value_counts().to_dict()
    distribusi_test = y_test.value_counts().to_dict()

    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    selisih_akurasi = round(train_accuracy - test_accuracy, 4)

    #Konfigurasi Parameter
    parameter_model = model.get_params()

    # Matrix korelasi fitur terhadap target
    corr_matrix = df.corr()

    plt.figure(figsize=(10,8))

    sns.heatmap(
        corr_matrix,
        annot=True,        # menampilkan semua nilai di setiap kotak
        cmap="coolwarm",
        fmt=".2f",         # 2 angka di belakang koma
        linewidths=0.5     # garis antar kotak
    )

    plt.title("Matriks Korelasi Antar Variabel")

    plt.tight_layout()
    plt.savefig("static/matriks_korelasi.png")
    plt.close()

    # =========================
    # EVALUASI MODEL
    # =========================
    y_pred = model.predict(X_test)
    
    estimators_range = [10, 20, 50, 100, 150, 200]
    oob_scores = []

    for n in estimators_range:
        model_temp = RandomForestClassifier(
            n_estimators=n,
            bootstrap=True,
            oob_score=True,
            random_state=42,
            class_weight="balanced"
        )
        model_temp.fit(X_train, y_train)
        oob_scores.append(model_temp.oob_score_)

    plt.figure()
    plt.plot(estimators_range, oob_scores)
    plt.xlabel("Jumlah Pohon (n_estimators)")
    plt.ylabel("OOB Score")
    plt.title("Pengaruh Jumlah Pohon terhadap OOB Score")
    plt.tight_layout()
    plt.savefig("static/grafik_jumlah_pohon.png")
    plt.close()

    # ======================
    # DATASET HASIL PREDIKSI RANDOM FOREST
    # ======================

    hasil_rf = X_test.copy()
    hasil_rf["Aktual"] = y_test.values
    hasil_rf["Prediksi"] = y_pred
    hasil_rf.to_csv("dataset_hasil_random_forest.csv", index=False)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # =========================
    # CEK OVERFITTING
    # =========================
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    selisih_akurasi = round(train_accuracy - test_accuracy, 4)

    if selisih_akurasi > 0.1:
        status_model = "Model mengalami overfitting"
    else:
        status_model = "Model cukup stabil"


    # =========================
    # FEATURE IMPORTANCE (FIX FINAL)
    # =========================

    # 🔥 Ambil nama fitur LANGSUNG dari model
    feature_names = model.feature_names_in_
    importances = model.feature_importances_

    # Gabungkan & urutkan (SATU SUMBER DATA)
    feature_sorted = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True
    )

    # Untuk tabel
    feature_data = [
        {
            "fitur": fitur,
            "nilai": round(nilai * 100, 2)
        }
        for fitur, nilai in feature_sorted
    ]
    
    # Untuk grafik (jika dipakai di template)
    fitur_grafik = [f for f, _ in feature_sorted]
    nilai_grafik = [round(v * 100, 2) for _, v in feature_sorted]

    # =========================
    # RANDOM FEATURE SELECTION
    # =========================
    max_features = model.max_features

    if max_features is None:
        max_features = "Semua Fitur"

    jumlah_fitur = X.shape[1]

    if max_features == "sqrt":
        fitur_dipilih = int(jumlah_fitur ** 0.5)
    elif max_features == "log2":
        import math
        fitur_dipilih = int(math.log2(jumlah_fitur))
    elif isinstance(max_features, int):
        fitur_dipilih = max_features
    else:
        fitur_dipilih = jumlah_fitur

    # =========================
    # VISUALISASI POHON - FULL DEPTH
    # =========================
    from sklearn.tree import plot_tree

    tree = model.estimators_[0]

    plt.figure(figsize=(24, 12))
    plot_tree(
        tree,
        feature_names=model.feature_names_in_,
        class_names=["Tidak Lulus", "Lulus"],
        filled=True,
        rounded=True,
        max_depth=3,   # 🔥 dibatasi
        fontsize=11
    )
    plt.title("Decision Tree (3 Level Pertama)", fontsize=16)
    plt.tight_layout()
    plt.savefig("static/contoh_pohon_3level.png", dpi=300)
    plt.close()

    # =========================
    # VISUALISASI POHON - 3 LEVEL
    # =========================
    plt.figure(figsize=(30, 15))
    plot_tree(
        tree,
        feature_names=model.feature_names_in_,
        class_names=["Tidak Lulus", "Lulus"],
        filled=True,
        rounded=True,
        fontsize=10
    )
    plt.title("Decision Tree (Full Depth)", fontsize=16)
    plt.tight_layout()
    plt.savefig("static/contoh_pohon_full.png", dpi=300)
    plt.close()

    # =========================
    # RENDER
    # =========================
    return render_template(
        "analisis.html",
        user=session["user"],
        role=session["role"],

        # Data mentah
        contoh_data_awal=contoh_data_awal,

        # Seleksi fitur
        contoh_X=contoh_X,
        contoh_y=contoh_y,
        jumlah_fitur=jumlah_fitur,
        jumlah_data=jumlah_data,

        # Preprocessing
        jumlah_awal=jumlah_awal,
        jumlah_bersih=jumlah_bersih,
        contoh_data=contoh_data,

        jumlah_train=jumlah_train,
        jumlah_test=jumlah_test,
        persen_train=persen_train,
        persen_test=persen_test,
        distribusi_train=distribusi_train,
        distribusi_test=distribusi_test,

        train_accuracy=round(train_accuracy,4),
        test_accuracy=round(test_accuracy,4),
        selisih_akurasi=selisih_akurasi,
        status_model=status_model,

        #Konfigurasi Parameter
        parameter_model=parameter_model,

        #Boostrap OOB Score
        oob_score=round(oob_score,4),

        max_features=max_features,
        fitur_dipilih=fitur_dipilih,

        # Evaluasi model
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,

        # Feature importance
        feature_data=feature_data,
        fitur_grafik=fitur_grafik,
        nilai_grafik=nilai_grafik
    )

# ======================
# PREDIKSI (ADMIN SAJA)
# ======================
@app.route("/prediksi", methods=["GET", "POST"])
def prediksi():
    if "user" not in session or session["role"] != "admin":
        return redirect(url_for("login"))

    if not session.get("model_trained"):
        flash("Model belum tersedia. Upload dataset terlebih dahulu.", "warning")
        return redirect(url_for("dashboard"))

    # Inisialisasi
    hasil = None         # 1=Lulus, 0=Tidak Lulus
    intervensi = ""
    status_adm = None     # 1=Lengkap, 0=Belum Lengkap

    if request.method == "POST":
        # Ambil data dari form
        rata = float(request.form["rata"])
        ukk = float(request.form["ukk"])
        prakerin = float(request.form["prakerin"])
        kehadiran = int(request.form["kehadiran"])
        sikap = int(request.form["sikap"])
        admin_input = int(request.form["admin"])  # 1=Lunas, 0=Belum

        data = [[rata, ukk, prakerin, kehadiran, sikap, admin_input]]

        # ======================
        # RULE WAJIB KELULUSAN (HYBRID SYSTEM)
        # ======================

        if (
            rata < 65 or
            ukk < 65 or
            prakerin < 65 or
            kehadiran < 70 or
            sikap < 2 or          # 2 = Cukup
            admin_input != 1      # 1 = Lunas
        ):
            hasil = 0  # Tidak Lulus
        else:
            # Jika semua syarat terpenuhi baru pakai Random Forest
            hasil_prediksi = model.predict(data)[0]
            hasil = int(hasil_prediksi)


        # ======================
        # INTERVENSI OTOMATIS (SINKRON DENGAN RULE)
        # ======================
        intervensi_list = []

        # --- INTERVENSI KRITIS (tidak memenuhi syarat kelulusan) ---
        if rata < 60:
            intervensi_list.append("Remedial nilai rata-rata (di bawah standar minimum)")

        if ukk < 60:
            intervensi_list.append("Remedial UKK (di bawah standar minimum)")

        if prakerin < 60:
            intervensi_list.append("Remedial Prakerin (di bawah standar minimum)")

        if kehadiran < 70:
            intervensi_list.append("Pembinaan kehadiran (di bawah 70%)")

        if sikap < 2:  # 1 = Kurang
            intervensi_list.append("Konseling sikap/disiplin")

        if admin_input != 1:
            intervensi_list.append("Lengkapi administrasi siswa")

        # --- INTERVENSI PEMBINAAN (masih lulus tapi mendekati batas) ---
        if 60 <= rata < 65:
            intervensi_list.append("Perlu peningkatan nilai rata-rata")

        if 60 <= ukk < 65:
            intervensi_list.append("Perlu peningkatan nilai UKK")

        if 60 <= prakerin < 65:
            intervensi_list.append("Perlu peningkatan nilai Prakerin")

        if 70 <= kehadiran < 80:
            intervensi_list.append("Perlu peningkatan kedisiplinan kehadiran")

        # Jika sistem menyatakan tidak lulus
        if hasil == 0:
            intervensi_list.append("Tindak lanjuti siswa secara intensif")

        if not intervensi_list:
            intervensi_list.append("Tidak diperlukan intervensi khusus")

        intervensi = "; ".join(intervensi_list)


        # ======================
        # Status administrasi (angka)
        # ======================
        status_adm = admin_input  # 1=Lengkap, 0=Belum Lengkap

        # ======================
        # Simpan ke CSV (quote agar aman)
        # ======================
        record = {
            "user": session["user"],
            "rata": rata,
            "ukk": ukk,
            "prakerin": prakerin,
            "kehadiran": kehadiran,
            "sikap": sikap,
            "admin": admin_input,
            "hasil": hasil,
            "intervensi": intervensi,
            "status_administrasi": status_adm,
            "waktu": datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        }

        file = "riwayat_prediksi.csv"
        df = pd.DataFrame([record])
        df.to_csv(
            file,
            mode="a",
            header=not os.path.exists(file),
            index=False,
            quoting=csv.QUOTE_NONNUMERIC
        )

    return render_template(
        "prediksi.html",
        user=session["user"],
        role=session["role"],
        hasil=hasil,
        intervensi=intervensi,
        status_adm=status_adm,
    )

# ======================
# Hasil Prediksi
# ======================
@app.route("/hasil-prediksi")
def hasil_prediksi():
    if "user" not in session:
        return redirect(url_for("login"))

    data = []
    file = "riwayat_prediksi.csv"
    if os.path.exists(file):
        df = pd.read_csv(file, quotechar='"')
        data = df.to_dict(orient="records")

    return render_template(
        "hasil_prediksi.html",
        data=data,
        role=session.get("role", "user"),
        user=session.get("user", "Guest")
    )



@app.route("/cetak-hasil-prediksi")
def cetak_hasil_prediksi():
    if "user" not in session or session["role"] != "admin":
        flash("Anda tidak memiliki hak akses mencetak", "danger")
        return redirect(url_for("hasil_prediksi"))

    import pandas as pd
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Table, TableStyle, Image, Spacer
    )
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_RIGHT
    from reportlab.lib import colors
    from datetime import datetime
    import os

    # ======================
    # FILE
    # ======================
    file_csv = "riwayat_prediksi.csv"
    file_pdf = "static/hasil_prediksi.pdf"
    logo_path = "static/images/logo_sekolah.png"

    df = pd.read_csv(file_csv)

    # ======================
    # DATA DINAMIS
    # ======================
    tanggal_cetak = datetime.now().strftime("%d %B %Y")
    nama_admin = session.get("user", "Admin")
    nomor_surat = "421.5/001/SMK/II/2026"

    nama_kepsek = "Antoni, M.Pd.T"
    nip_kepsek = "19710408 199512 1 001"

    # ======================
    # DOKUMEN
    # ======================
    doc = SimpleDocTemplate(
        file_pdf,
        pagesize=landscape(A4),
        rightMargin=40,
        leftMargin=30,
        topMargin=30,
        bottomMargin=30
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleStyle",
        fontSize=16,
        alignment=TA_CENTER,
        fontName="Helvetica-Bold"
    )

    center_small = ParagraphStyle(
    "CenterSmall",
    fontSize=11,      
    leading=14,         
    alignment=TA_CENTER
    )


    right_style = ParagraphStyle(
        "RightStyle",
        fontSize=9,
        alignment=TA_RIGHT
    )

    wrap_style = ParagraphStyle(
        "Wrap",
        fontSize=8,
        leading=10,
        wordWrap="CJK"
    )

    elements = []

    # ======================
    # KOP SURAT
    # ======================
    logo = Image(logo_path, 60, 60) if os.path.exists(logo_path) else ""

    kop = Table([
        [
            logo,
            Paragraph(
                """
                <b>SMK NEGERI 1 GUGUAK</b><br/>
                Guguak VIII Koto, Kec. Guguak, Kabupaten Lima Puluh Kota, Sumatera Barat 26253<br/>
                Telp. (021) 12345678 | Email: info@smk1guguak.sch.id
                """,
                center_small
            )

        ]
    ], colWidths=[70, 650])
    rowHeights=[30, 28]

    kop.setStyle(TableStyle([
        ("TOPPADDING", (0,0), (-1,-1), 0),
        ("BOTTOMPADDING", (0,0), (-1,-1), 0),
        ("LEFTPADDING", (0,0), (-1,-1), 0),
        ("RIGHTPADDING", (0,0), (-1,-1), 0),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))


    elements.append(kop)
    elements.append(
    Table(
        [[""]],
        colWidths=[720],
        rowHeights=[3],  # tinggi baris kecil → garis nempel
        style=TableStyle([
            ("LINEABOVE", (0,0), (-1,-1), 0.8, colors.black),
            ("LINEBELOW", (0,0), (-1,-1), 2.2, colors.black),
        ])
    )
)

    elements.append(Spacer(1, 10))

    # ======================
    # NOMOR SURAT
    # ======================
    elements.append(
        Paragraph(f"Nomor : {nomor_surat}", right_style)
    )

    elements.append(Spacer(1, 10))

    # ======================
    # JUDUL
    # ======================
    elements.append(
        Paragraph("LAPORAN HASIL PREDIKSI KELULUSAN SISWA SMK", title_style)
    )

    elements.append(Spacer(1, 12))

    # ======================
    # TABEL DATA
    # ======================
    table_data = [[
        "No", "Rata-rata", "UKK", "Prakerin",
        "Kehadiran", "Sikap", "Administrasi",
        "Hasil", "Intervensi", "Waktu"
    ]]

    for i, row in df.iterrows():
        table_data.append([
            i + 1,
            row["rata"],
            row["ukk"],
            row["prakerin"],
            row["kehadiran"],
            row["sikap"],
            "Lengkap" if row["admin"] == 1 else "Belum Lengkap",
            "Lulus" if row["hasil"] == 1 else "Tidak Lulus",
            Paragraph(str(row["intervensi"]), wrap_style),
            row["waktu"]
        ])

    table = Table(
        table_data,
        colWidths=[30, 55, 45, 55, 60, 40, 70, 55, 230, 90]
    )

    table.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 1, colors.black),
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("FONTSIZE", (0,0), (-1,-1), 8),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 20))

    # ======================
    # TANDA TANGAN
    # ======================
    ttd = Table([
        [
            "",
            Paragraph(
                f"""
                {tanggal_cetak}<br/><br/>
                Kepala Sekolah<br/><br/><br/><br/>
                <b>{nama_kepsek}</b><br/>
                NIP. {nip_kepsek}
                """,
                center_small
            )
        ]
    ], colWidths=[500, 220])

    elements.append(ttd)

    # ======================
    # FOOTNOTE
    # ======================
    elements.append(Spacer(1, 10))
    elements.append(
        Paragraph(f"Dicetak oleh: {nama_admin}", styles["Normal"])
    )

    doc.build(elements)

    return redirect(url_for("static", filename="hasil_prediksi.pdf"))

# ======================
# Hapus Hasil
# ======================
@app.route("/hapus-hasil-prediksi", methods=["POST"])
def hapus_hasil_prediksi():
    if "user" not in session or session["role"] != "admin":
        return redirect(url_for("login"))

    try:
        import os
        if os.path.exists("riwayat_prediksi.csv"):
            os.remove("riwayat_prediksi.csv")

        flash("Semua hasil prediksi berhasil dihapus.", "success")
    except Exception as e:
        flash(f"Gagal menghapus data: {str(e)}", "danger")

    return redirect(url_for("hasil_prediksi"))


# ======================
# LOGOUT
# ======================
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))
# ======================
# RUN (LOCAL ONLY)
# ======================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)



