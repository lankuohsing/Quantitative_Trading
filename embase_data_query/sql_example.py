import oracledb

USER = "indsrvdata_r"
PASS = "ghj*EFB"
HOST = "61.129.116.184"
PORT = 11521
SERVICE_NAME = "EMBASE"

# 方式A：Easy Connect 字符串（最直观）
dsn = f"{HOST}:{PORT}/{SERVICE_NAME}"

try:
    with oracledb.connect(user=USER, password=PASS, dsn=dsn) as conn:
        with conn.cursor() as cur:
            # 1）先确认能连上
            cur.execute("SELECT 1 FROM DUAL")
            print("✅ 连接成功，DUAL 返回：", cur.fetchone())

            # 2）查你要的表（限制行数，避免拉爆）
            sql = """
                SELECT *
                FROM NEWSADMIN.SPTM_MARKETRELATION
                WHERE ROWNUM <= 20
            """
            cur.execute(sql)
            rows = cur.fetchall()

            # 打印列名
            cols = [d[0] for d in cur.description]
            print("列：", cols)
            for r in rows:
                print(r)

except Exception as e:
    print("❌ 出错：", e)