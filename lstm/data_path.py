__author__ = 'yuhongliang324'
import os


dn = os.path.dirname(os.path.abspath(__file__))
dn = os.path.join(dn, '..')

accident_data_root = os.path.join(dn, 'permutation/accident')
accident_train_root = os.path.join(accident_data_root, 'train')
accident_test_root = os.path.join(accident_data_root, 'test')

earthquake_data_root = os.path.join(dn, 'permutation/earthquake')
earthquake_train_root = os.path.join(earthquake_data_root, 'train')
earthquake_test_root = os.path.join(earthquake_data_root, 'test')
