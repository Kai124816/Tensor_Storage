import csv
import sys

#Rounds Each Entry to the third decimal point
def round_csv(file: str) -> None:
    with open(file, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)

    new_rows = []
    for i, row in enumerate(rows):
        if i == 0:
            new_rows.append(row)
            continue
        new_row = []
        for item in row:
            if item.strip() == '':
                new_row.append(item)
            else:
                try:
                    val = float(item)
                    if '.' in item or 'e' in item.lower():
                        new_row.append(f"{val:.3f}")
                    else:
                        new_row.append(item)
                except ValueError:
                    new_row.append(item)
        new_rows.append(new_row)

    with open(file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(new_rows)

#Removes column from CSV file
def remove_column(file: str, column_name: str) -> None:
    with open(file, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
    
    if not rows:
        return
        
    header = rows[0]
    if column_name not in header:
        return
        
    col_idx = header.index(column_name)
    
    new_rows = []
    for row in rows:
        if len(row) > col_idx:
            new_row = row[:col_idx] + row[col_idx+1:]
            new_rows.append(new_row)
        else:
            new_rows.append(row)
            
    with open(file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(new_rows)

#Sums up two columns and turns each column into a percentage
def consolidate_to_percentage(file: str, col_1: str, col_2: str) -> None:
    with open(file, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
        
    if not rows:
        return
        
    header = rows[0]
    if col_1 not in header or col_2 not in header:
        return
        
    idx_1 = header.index(col_1)
    idx_2 = header.index(col_2)
    
    new_rows = [header]
    for row in rows[1:]:
        new_row = list(row)
        if len(row) > max(idx_1, idx_2):
            val_1_str = row[idx_1].strip()
            val_2_str = row[idx_2].strip()
            try:
                val_1 = float(val_1_str) if val_1_str else 0.0
                val_2 = float(val_2_str) if val_2_str else 0.0
                total = val_1 + val_2
                
                if total != 0:
                    pct_1 = (val_1 / total) * 100.0
                    pct_2 = (val_2 / total) * 100.0
                    new_row[idx_1] = f"{pct_1:.3f}%"
                    new_row[idx_2] = f"{pct_2:.3f}%"
                else:
                    new_row[idx_1] = "0.000%"
                    new_row[idx_2] = "0.000%"
            except ValueError:
                pass
        new_rows.append(new_row)
        
    with open(file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(new_rows)

def main():
    if len(sys.argv) < 3:
        raise ValueError("Need to pass in at least two arguments option and file")

    option = sys.argv[1]
    filename = sys.argv[2]
    if option == "round":
        round_csv(filename)
    elif option == "remove":
        if len(sys.argv) != 4:
            raise ValueError("need to pass in column to remove")
            return
        column = sys.argv[3]
        remove_column(filename, column)
    elif option == "consolidate":
        if len(sys.argv) != 5:
            raise ValueError("need to pass in two columns to consolidate")
            return
        col_1 = sys.argv[3]
        col_2 = sys.argv[4]
        consolidate_to_percentage(filename, col_1, col_2)
        
    else:
        raise ValueError("Invalid Option")
        return
        

if __name__ == "__main__":
    main()