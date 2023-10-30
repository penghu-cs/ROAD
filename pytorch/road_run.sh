# Office-31

python new_dual_noise_train_image_office.py --gpu_id 3 --net ResNet50 --dset office --s_dset_path ../data/office/amazon_list_40.txt --w_dset_path ../data/office/amazon_list.txt  --t_dset_path ../data/office/dslr_list.txt --warm_up_epoch 1800;
python new_dual_noise_train_image_office.py --gpu_id 3 --net ResNet50 --dset office --s_dset_path ../data/office/amazon_list_40.txt --w_dset_path ../data/office/amazon_list.txt  --t_dset_path ../data/office/webcam_list.txt --warm_up_epoch 1800;
python new_dual_noise_train_image_office.py --gpu_id 3 --net ResNet50 --dset office --s_dset_path ../data/office/webcam_list_40.txt --w_dset_path ../data/office/webcam_list.txt  --t_dset_path ../data/office/amazon_list.txt --warm_up_epoch 900;
python new_dual_noise_train_image_office.py --gpu_id 3 --net ResNet50 --dset office --s_dset_path ../data/office/webcam_list_40.txt --w_dset_path ../data/office/webcam_list.txt  --t_dset_path ../data/office/dslr_list.txt --warm_up_epoch 900;
python new_dual_noise_train_image_office.py --gpu_id 3 --net ResNet50 --dset office --s_dset_path ../data/office/dslr_list_40.txt --w_dset_path ../data/office/dslr_list.txt  --t_dset_path ../data/office/webcam_list.txt --warm_up_epoch 800;
python new_dual_noise_train_image_office.py --gpu_id 3 --net ResNet50 --dset office --s_dset_path ../data/office/dslr_list_40.txt --w_dset_path ../data/office/dslr_list.txt  --t_dset_path ../data/office/amazon_list.txt --warm_up_epoch 800;

# Office-Home

python new_dual_noise_train_image_officehome.py --gpu_id 0 --net ResNet101 --dset office-home --s_dset_path ../data/office-home/Art_40_.txt --w_dset_path ../data/office-home/Art.txt --t_dset_path ../data/office-home/Clipart_shared.txt --warm_up_epoch 650;
python new_dual_noise_train_image_officehome.py --gpu_id 0 --net ResNet101 --dset office-home --s_dset_path ../data/office-home/Art_40_.txt --w_dset_path ../data/office-home/Art.txt --t_dset_path ../data/office-home/Product_shared.txt --warm_up_epoch 650;
python new_dual_noise_train_image_officehome.py --gpu_id 0 --net ResNet101 --dset office-home --s_dset_path ../data/office-home/Art_40_.txt --w_dset_path ../data/office-home/Art.txt --t_dset_path ../data/office-home/Real_World_shared.txt --warm_up_epoch 650;
python new_dual_noise_train_image_officehome.py --gpu_id 0 --net ResNet101 --dset office-home --s_dset_path ../data/office-home/Clipart_40_.txt --w_dset_path ../data/office-home/Clipart.txt --t_dset_path ../data/office-home/Art_shared.txt --warm_up_epoch 1050;
python new_dual_noise_train_image_officehome.py --gpu_id 0 --net ResNet101 --dset office-home --s_dset_path ../data/office-home/Clipart_40_.txt --w_dset_path ../data/office-home/Clipart.txt --t_dset_path ../data/office-home/Product_shared.txt --warm_up_epoch 1050;
python new_dual_noise_train_image_officehome.py --gpu_id 0 --net ResNet101 --dset office-home --s_dset_path ../data/office-home/Clipart_40_.txt --w_dset_path ../data/office-home/Clipart.txt --t_dset_path ../data/office-home/Real_World_shared.txt --warm_up_epoch 1050;
python new_dual_noise_train_image_officehome.py --gpu_id 0 --net ResNet101 --dset office-home --s_dset_path ../data/office-home/Product_40_.txt --w_dset_path ../data/office-home/Product.txt --t_dset_path ../data/office-home/Art_shared.txt --warm_up_epoch 800;
python new_dual_noise_train_image_officehome.py --gpu_id 0 --net ResNet101 --dset office-home --s_dset_path ../data/office-home/Product_40_.txt --w_dset_path ../data/office-home/Product.txt --t_dset_path ../data/office-home/Clipart_shared.txt --warm_up_epoch 800;
python new_dual_noise_train_image_officehome.py --gpu_id 0 --net ResNet101 --dset office-home --s_dset_path ../data/office-home/Product_40_.txt --w_dset_path ../data/office-home/Product.txt --t_dset_path ../data/office-home/Real_World_shared.txt --warm_up_epoch 800;
python new_dual_noise_train_image_officehome.py --gpu_id 0 --net ResNet101 --dset office-home --s_dset_path ../data/office-home/Real_World_40_.txt --w_dset_path ../data/office-home/Real_World.txt --t_dset_path ../data/office-home/Art_shared.txt --warm_up_epoch 700;
python new_dual_noise_train_image_officehome.py --gpu_id 0 --net ResNet101 --dset office-home --s_dset_path ../data/office-home/Real_World_40_.txt --w_dset_path ../data/office-home/Real_World.txt --t_dset_path ../data/office-home/Clipart_shared.txt --warm_up_epoch 700;
python new_dual_noise_train_image_officehome.py --gpu_id 0 --net ResNet101 --dset office-home --s_dset_path ../data/office-home/Real_World_40_.txt --w_dset_path ../data/office-home/Real_World.txt --t_dset_path ../data/office-home/Product_shared.txt --warm_up_epoch 700;