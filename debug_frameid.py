# debug_frameid.py
import pandas as pd
df_gt = pd.read_csv('dump_openface.csv', header=None, names=['FrameID','CP','Idx','Val'])
df_tg = pd.read_csv('dump_recover.csv', header=None, names=['FrameID','CP','Idx','Val'])

# CP4 passed - let's see the FrameID ranges
cp4_gt = df_gt[df_gt['CP']=='CP4']
cp4_tg = df_tg[df_tg['CP']=='CP4']

print('CP4 GT FrameID range:', cp4_gt['FrameID'].min(), '-', cp4_gt['FrameID'].max())
print('CP4 TG FrameID range:', cp4_tg['FrameID'].min(), '-', cp4_tg['FrameID'].max())

# CP1 ranges  
cp1_gt = df_gt[df_gt['CP']=='CP1']
cp1_tg = df_tg[df_tg['CP']=='CP1']
print('CP1 GT FrameID range:', cp1_gt['FrameID'].min(), '-', cp1_gt['FrameID'].max())
print('CP1 TG FrameID range:', cp1_tg['FrameID'].min(), '-', cp1_tg['FrameID'].max())
