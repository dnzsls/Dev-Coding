# =============================================================================
# GERÇEK ÇALIŞAN vs LP KARŞILAŞTIRMASI - TEK AKIŞ
# =============================================================================

import pandas as pd
import numpy as np
from scipy.optimize import linprog

# =============================================================================
# 1. VERİ YÜKLEME
# =============================================================================

# Shift tanımları
df_shifts = pd.read_excel('vardiyalar.xlsx')

# Gerçek çalışan verisi
df_actual = pd.read_excel('gercek_calisanlar.xlsx')

print(f"Shift sayısı: {len(df_shifts)}")
print(f"Gerçek çalışan satır: {len(df_actual)}")

# Kuyruk-Company kuralları
queue_company_rules = {
    'a_cagrilari': ['inhouse', 'outsource'],
    'b_cagrilari': ['inhouse'],
    'c_cagrilari': ['inhouse']
}


# =============================================================================
# 2. YARDIMCI FONKSİYONLAR
# =============================================================================

def is_slot_in_shift(slot, start, end):
    """Slot bu shift içinde mi?"""
    if start <= end:
        return start <= slot < end
    else:  # Gece shift (22:00-06:00)
        return slot >= start or slot < end


def create_shift_coverage_matrix(df_shifts):
    """Her shift'in hangi slotları kapsadığını hesapla"""
    shift_coverage = {}
    
    for _, row in df_shifts.iterrows():
        shift = row['shift']
        start = str(row['start'])[:5]
        end = str(row['end'])[:5]
        company = row['company']
        
        key = f"{shift}_{company}"
        
        slots = []
        for h in range(24):
            for m in ['00', '30']:
                slot = f"{h:02d}:{m}"
                if is_slot_in_shift(slot, start, end):
                    slots.append(slot)
        
        shift_coverage[key] = {
            'shift': shift,
            'company': company,
            'start': start,
            'end': end,
            'slots': slots
        }
    
    return shift_coverage


def calculate_shift_slot_matrix(shift_coverage):
    """Shift-Slot binary matrisi"""
    slots = [f"{h:02d}:{m}" for h in range(24) for m in ['00', '30']]
    shifts = list(shift_coverage.keys())
    
    matrix = pd.DataFrame(0, index=slots, columns=shifts)
    
    for shift_key, info in shift_coverage.items():
        for slot in info['slots']:
            matrix.loc[slot, shift_key] = 1
    
    return matrix


def calculate_actual_by_slot(df_actual, date, queue):
    """Belirli gün ve kuyruk için slot bazında gerçek çalışan"""
    df_day = df_actual[df_actual['working_date'] == date]
    df_day = df_day[df_day['line_based_main_group'] == queue]
    
    slots = [f"{h:02d}:{m}" for h in range(24) for m in ['00', '30']]
    actual_by_slot = {}
    
    for slot in slots:
        count = 0
        for _, row in df_day.iterrows():
            start = str(row['shifts_start_hour'])[:5]
            end = str(row['shifts_end_hour'])[:5]
            
            if is_slot_in_shift(slot, start, end):
                count += row['calisan_kisi_sayisi']
        
        actual_by_slot[slot] = count
    
    return actual_by_slot


def optimize_shift_distribution(need_by_slot, df_shift_matrix):
    """LP ile shift dağılımı optimize et"""
    slots = df_shift_matrix.index.tolist()
    shifts = df_shift_matrix.columns.tolist()
    
    active_slots = [s for s in slots if s in need_by_slot and need_by_slot[s] > 0]
    
    if not active_slots:
        return {shift: 0 for shift in shifts}
    
    # Amaç: Toplam kişiyi minimize et
    c = [1] * len(shifts)
    
    # Kısıtlar
    A_ub = []
    b_ub = []
    
    for slot in active_slots:
        constraint = [-df_shift_matrix.loc[slot, shift] for shift in shifts]
        A_ub.append(constraint)
        b_ub.append(-need_by_slot[slot])
    
    bounds = [(0, 100) for _ in shifts]
    
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if result.success:
        assignments = {}
        for i, shift in enumerate(shifts):
            assignments[shift] = int(np.ceil(result.x[i]))
        return assignments
    else:
        print(f"Optimizasyon başarısız: {result.message}")
        return None


# =============================================================================
# 3. KARŞILAŞTIRMA FONKSİYONLARI
# =============================================================================

def compare_slots_all_days(df_actual, df_shifts, queue, allowed_companies):
    """Tüm günler için slot bazlı detay + shift atamaları"""
    
    # Shift coverage (company filtreli)
    df_shifts_filtered = df_shifts[df_shifts['company'].isin(allowed_companies)]
    shift_coverage = create_shift_coverage_matrix(df_shifts_filtered)
    df_shift_matrix = calculate_shift_slot_matrix(shift_coverage)
    
    dates = df_actual[df_actual['line_based_main_group'] == queue]['working_date'].unique()
    
    print(f"Kuyruk: {queue}")
    print(f"İzin verilen company: {allowed_companies}")
    print(f"Kullanılabilir shift: {len(shift_coverage)}")
    print(f"Toplam gün: {len(dates)}")
    
    all_slot_rows = []
    all_shift_rows = []
    
    for date in dates:
        # Gerçek çalışan (slot bazında)
        actual_by_slot = calculate_actual_by_slot(df_actual, date, queue)
        
        # LP çözümü
        lp_assignments = optimize_shift_distribution(actual_by_slot, df_shift_matrix)
        
        if lp_assignments is None:
            continue
        
        # LP'den slot bazında hesapla
        lp_by_slot = {}
        for slot in df_shift_matrix.index:
            total = 0
            for shift, count in lp_assignments.items():
                if df_shift_matrix.loc[slot, shift] == 1:
                    total += count
            lp_by_slot[slot] = total
        
        # Slot satırları
        for slot in sorted(actual_by_slot.keys()):
            actual = actual_by_slot.get(slot, 0)
            lp = lp_by_slot.get(slot, 0)
            
            if actual > 0 or lp > 0:
                all_slot_rows.append({
                    'tarih': date,
                    'slot': slot,
                    'gercek_calisan': int(actual),
                    'lp_atama': int(lp),
                    'fark': int(lp - actual)
                })
        
        # Shift atamaları
        for shift_key, count in lp_assignments.items():
            if count > 0:
                info = shift_coverage[shift_key]
                all_shift_rows.append({
                    'tarih': date,
                    'shift_kod': info['shift'],
                    'company': info['company'],
                    'baslangic': info['start'],
                    'bitis': info['end'],
                    'lp_kisi_sayisi': int(count)
                })
    
    df_slots = pd.DataFrame(all_slot_rows).sort_values(['tarih', 'slot']).reset_index(drop=True)
    df_shifts_result = pd.DataFrame(all_shift_rows).sort_values(['tarih', 'baslangic']).reset_index(drop=True)
    
    return df_slots, df_shifts_result, shift_coverage


def create_summary(df_slots):
    """Gün bazlı özet"""
    summary = df_slots.groupby('tarih').agg({
        'gercek_calisan': 'sum',
        'lp_atama': 'sum',
        'fark': 'sum'
    }).reset_index()
    
    summary['tasarruf_pct'] = round((summary['gercek_calisan'] - summary['lp_atama']) / summary['gercek_calisan'] * 100, 1)
    
    return summary


# =============================================================================
# 4. ÇALIŞTIR
# =============================================================================

queue = 'a_cagrilari'

# Karşılaştırma yap
df_slots, df_shifts_lp, shift_coverage = compare_slots_all_days(
    df_actual, 
    df_shifts, 
    queue, 
    queue_company_rules[queue]
)

# Özet
df_summary = create_summary(df_slots)

print(f"\n{'='*60}")
print("GÜN BAZLI ÖZET")
print(f"{'='*60}")
print(df_summary)

print(f"\n{'='*60}")
print("GENEL ÖZET")
print(f"{'='*60}")
print(f"Ortalama gerçek: {df_summary['gercek_calisan'].mean():.0f}")
print(f"Ortalama LP: {df_summary['lp_atama'].mean():.0f}")
print(f"Ortalama fark: {df_summary['fark'].mean():.0f}")
print(f"Ortalama tasarruf: %{df_summary['tasarruf_pct'].mean():.1f}")

print(f"\n{'='*60}")
print("SLOT BAZLI DETAY (İlk 20 satır)")
print(f"{'='*60}")
print(df_slots.head(20))

print(f"\n{'='*60}")
print("LP SHIFT ATAMALARI (İlk 20 satır)")
print(f"{'='*60}")
print(df_shifts_lp.head(20))

# Excel'e kaydet
# df_slots.to_excel('slot_karsilastirma.xlsx', index=False)
# df_shifts_lp.to_excel('lp_shift_atamalari.xlsx', index=False)
# df_summary.to_excel('gun_ozet.xlsx', index=False)
