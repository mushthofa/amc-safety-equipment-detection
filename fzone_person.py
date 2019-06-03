# 23-09-2019
# Azure Czure

# -- CHANGE THIS --
test_path = "frank.mp4"
# Start:
fz_poly1 = [(0.55, 0.48), (0.79, 0.48), (1.24, 0.59), (0.85, 0.59)] # (x1, y1), (x2, y2), ...
# End:
fz_poly2 = [(0.25, 0.48), (0.39, 0.48), (0.74, 0.59), (0.35, 0.59)] # (x1, y1), (x2, y2), ...
rotate_right = True
skip_cnt = 50
frame_step = 10
cname = 'person'

exec(open('forbidden_zone.py').read())
