set -v
python label_centerdeg.py  # label, imgw, imgh, x, y, w, h, centerx, centery, 360deg
python cheby_fit.py  # label, imgw, imgh, x, y, w, h, centerx, centery, coef(16,24)
python train_coef_xml.py
python label_polygon.py
python val_polygon_xml.py
mkdir ../data/
ln -s ../sbd ../data/VOCsbdche
#rsync -av sbd_ESESEG_ImageSets/ ../data/VOCsbdche/ImageSets/Segmentation/
