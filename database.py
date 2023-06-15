import sqlite3
import datetime
import requests, json
import mes_requests as mes
def change_date_format(str_time):
	if len(str(str_time))==2 :
		return str(str_time)
	else :
		return '0'+str(str_time)
def change_text(str_num):
    return int(str_num.replace(",", ""))
def get_datetime():
    now = datetime.datetime.now()
    return f"{now.year}-{change_date_format(now.month)}-{change_date_format(now.day)} {change_date_format(now.hour)}:{change_date_format(now.minute)}:{change_date_format(now.second)}"
def get_date():
    now = datetime.datetime.now()
    return f"{now.year}-{change_date_format(now.month)}-{change_date_format(now.day)}"


        
        
class SQLDatabase():
    def __init__(self):
        self.con = sqlite3.connect("./mj_db.db", check_same_thread=False)
        self.cur = self.con.cursor()
        self.create_table()

    def create_table(self):
        try:
            self.cur.execute('''CREATE TABLE detect(id INTEGER PRIMARY KEY AUTOINCREMENT, start_date TEXT, 
                update_date TEXT, end_date TEXT, workorder_item_id INTEGER, 
                inspection_date TEXT, inspector_name TEXT, workorder_quantity INTEGER, 
                inspection_quatity INTEGER, inspection_percent INTEGER, bad_quantity INTEGER, 
                bad_type TEXT, normal_quantity INTEGER, workorder TEXT, commit_table INTEGER);
                ''')
            self.con.commit()
        except sqlite3.OperationalError:
            print("already exists")
    def check_post_data(self):
        select_table = self.select_table()
        enddate_inspection = self.select_enddate_table()
        print(enddate_inspection)
        for end_ins in enddate_inspection:
            print(end_ins[0])
            inspection_json = {
                "workorder_item_id": end_ins[4], 
                "inspection_date": end_ins[5], 
                "start_date":end_ins[1],
                "end_date":end_ins[2],
                "inspector_name":end_ins[6],
                "workorder_quantity":end_ins[7],
                "inspection_quatity":end_ins[8],
                "inspection_percent":end_ins[9],
                "bad_quantity":end_ins[10],
                "bad_type":[{"143": int(end_ins[11].split(",")[0]), "116": int(end_ins[11].split(",")[1]), "115":int(end_ins[11].split(",")[2])}],
                "normal_quantity":end_ins[12],       
            }
            if mes.post_inspection(inspection_json) == 200:
                self.update_post_table(end_ins[2], end_ins[0])
            print(inspection_json)
    def check_commit_data(self):
        today = get_date()

    def check_today_table(self, inspection_json, workorder):
        today = get_date()
        
        today_inspection = self.select_today_table()
        print(today_inspection)
        if not today_inspection :
            self.insert_table(inspection_json, workorder)
            today_inspection = self.select_today_table()
            return today_inspection[0]
        else:
            return today_inspection[0]

    def insert_table(self, inspection_json, workorder):
        update_date = get_datetime()
        print(inspection_json, workorder)
        bad_type = f'{inspection_json["bad_type"][0]["143"]},{inspection_json["bad_type"][0]["116"]},{inspection_json["bad_type"][0]["115"]}'
        print(bad_type)
        sql = '''INSERT INTO detect (start_date, update_date, end_date, workorder_item_id, inspection_date, inspector_name, workorder_quantity, inspection_quatity, inspection_percent, bad_quantity, bad_type, normal_quantity, workorder, commit_table) 
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)'''
        task = (inspection_json["start_date"],update_date,inspection_json["end_date"],
        inspection_json["workorder_item_id"],inspection_json["inspection_date"],inspection_json["inspector_name"],
        inspection_json["workorder_quantity"],inspection_json["inspection_quantity"],inspection_json["inspection_percent"],
        inspection_json["bad_quantity"],bad_type,inspection_json["normal_quantity"], workorder, 0)
            
        self.cur.execute(sql, task)
        self.con.commit()
    def insert_example_table(self):
        update_date = get_datetime()
        bad_type = f'0,0,0'
        sql = '''INSERT INTO detect (start_date, update_date, end_date, workorder_item_id, inspection_date, 
        inspector_name, workorder_quantity, inspection_quatity, inspection_percent, bad_quantity, bad_type, normal_quantity, workorder, commit_table) 
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)'''
        task = ("2023-04-03 09:00:00","2023-04-03 15:00:00",'',
        60,"2023-04-03","김명수",
        10002,10002,50,
        1,bad_type,10002, 'WK-20230410-001', 0)
        
        self.cur.execute(sql, task)
        self.con.commit()
        
    def update_post_table(self, end_date, id):

        self.cur.execute('''UPDATE detect SET end_date = ? WHERE id = ?;''', (end_date, id,))
        self.con.commit()
    
    def update_commit_table(self, id):

        self.cur.execute('''UPDATE detect SET commit_table = ? WHERE id = ?;''', (1, id,))
        self.con.commit()

    def update_table(self, inspection_json, id, workorder):
        update_date = get_datetime()
        bad_type = f'{inspection_json["bad_type"][0]["143"]},{inspection_json["bad_type"][0]["116"]},{inspection_json["bad_type"][0]["115"]}'
        self.cur.execute('''UPDATE detect SET start_date = ?, update_date = ?, end_date = ?, workorder_item_id = ?,
        inspection_date = ?, inspector_name = ?, workorder_quantity = ?, inspection_quatity = ?, inspection_percent = ?, 
        bad_quantity = ?, bad_type = ?, normal_quantity = ?, workorder = ? WHERE id = ?''',
        (inspection_json["start_date"], update_date, inspection_json["end_date"],
        inspection_json["workorder_item_id"], inspection_json["inspection_date"], inspection_json["inspector_name"],
        inspection_json["workorder_quantity"], inspection_json["inspection_quantity"], inspection_json["inspection_percent"],
        inspection_json["bad_quantity"], bad_type, inspection_json["normal_quantity"],workorder, id,))

        self.con.commit()

    def select_commit_table(self, workorder_item_id, workorder):
        today = get_date()
        self.cur.execute("SELECT * FROM detect WHERE workorder_item_id = ? and workorder = ? and inspection_date != ? and commit_table = 0", (workorder_item_id, workorder, today,))
        row = self.cur.fetchall()
        return row
    
    def select_enddate_table(self):
        today = get_date()
        self.cur.execute("SELECT * FROM detect WHERE end_date = '' and inspection_date != ? ", (today,))
        row = self.cur.fetchall()
        return row

    def select_table(self):

        self.cur.execute("SELECT * FROM detect ")
        row = self.cur.fetchall()

        print(row)
        return row

    def select_today_table(self):
        today = get_date()
        self.cur.execute("SELECT * FROM detect WHERE inspection_date = ? and end_date = '' ;", (today,))
        row = self.cur.fetchall()
        return row

    def close_db(self):
        self.con.close()