# 23-09-2019
# Azure Czure

# -- CHANGE THIS --
test_path = "bikey.mp4"
# Start:
fz_poly1 = [(0.0, 0.85), (0.3, 0.5), (0.55, 0.7), (0.4, 1.0), (0.0, 1.0)] # (x1, y1), (x2, y2), ...
# End:
fz_poly2 = [(0.0, 0.85), (0.3, 0.5), (0.55, 0.7), (0.4, 1.0), (0.0, 1.0)] # (x1, y1), (x2, y2), ...
rotate_right = False
skip_cnt = 100
frame_step = 5
cname = 'bicycle'

exec(open('forbidden_zone.py').read())
