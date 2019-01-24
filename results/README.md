### HR-Net
The prediction npy files and dataset csv files. There is my ```./results```:  
```
results/
├── 1.jpg
├── berk.csv
├── berk-test.csv
├── GoogLeNet_binary61_feature_imagenet_all.npy
├── GoogLeNet_binary61_feature_imagenet_test7600.npy
├── googlenet_weights.npy
├── logs
│   ├── log1.png
│   ├── models_log.png
│   └── new_models_log.png
├── metrics
│   ├── binary51_imagenet_acc7600_RAP.csv
│   ├── binary51_imagenet_ma7600_RAP.csv
│   ├── binary61_50_imagenet_acc7600.csv
│   ├── binary61_acc7600.csv
│   ├── binary61_acc.csv
│   ├── binary61_epoch50_acc7600.csv
│   └── binary61_imagenet_acc7600.csv
├── models
│   ├── GoogleLenet.png
│   ├── GoogleLenet-SPP.png
│   ├── hiarBayesGoogleLenet.png
│   ├── hiarGoogleLenet.png
│   ├── hiarGoogleLenet-SPP.png
│   ├── hiarGoogleLenet-WAM.png
│   ├── miniGoogleLenet.png
│   ├── OE-DC-GoogleLenet-SPP-lower.png
│   └── OE-DC-GoogleLenet-SPP.png
├── myRAP_labels_pd.csv
├── OEDCWPAL_PETA_features_all.pickle
├── partGoogleLenet.png
├── PETA_coarse_to_fine_labels_pd.csv
├── PETA.csv
├── PETA_labels_pd.csv
├── PETA_lowerBody_labels_pd.csv
├── PETA_ratio_positive_samples_for_attributes.json
├── predictions
│   ├── app.npy
│   ├── GoogLeNet_binary3_binary3_b2-32_lr0.0002_final_model_predictions_imagenet_test_RAP.npy
│   ├── GoogLeNet_binary51_binary51_b2_lr0.0002_1000_final_model_predictions_imagenet_test_RAP.npy
│   ├── GoogLeNet_binary51_binary51_b2_lr0.0002_160*75_final_model_predictions_imagenet_test_RAP.npy
│   ├── GoogLeNet_binary51_binary51_b2_lr0.0002_final_model_predictions_imagenet_test_RAP.npy
│   ├── GoogLeNet_binary51_binary51_b2_lr0.0002_lossweight_final_model_predictions_imagenet_test_RAP.npy
│   ├── GoogLeNet_binary51_binary51_b2_lr0.0002_nobn_final_model_predictions_imagenet_test_RAP.npy
│   ├── GoogLeNet_binary51_binary51_b4_mysplit_final_model_predictions_imagenet_test_RAP.npy
│   ├── GoogLeNet_binary51_binary51_final_model_predictions_imagenet_test_RAP.npy
│   ├── GoogLeNet_binary61_predictions50_imagenet_test7600.npy
│   ├── GoogLeNet_binary61_predictions50_test7600.npy
│   ├── GoogLeNet_binary61_predictions_imagenet_all.npy
│   ├── GoogLeNet_binary61_predictions_imagenet_test7600.npy
│   ├── GoogLeNet_binary61_predictions_test7600.npy
│   ├── GoogLeNet_binary61_predictions_test.npy
│   ├── GoogLeNet_binary61_X_test.npy
│   ├── GoogLeNet_binary61_y_test.npy
│   ├── GoogLeNet_binary9-depth9_predictions_imagenet_berk.npy
│   ├── GoogLeNet_depth1_binary61_predictions50_imagenet_test7600.npy
│   ├── GoogLeNet_depth2_binary61_predictions50_imagenet_test7600.npy
│   ├── GoogLeNet_depth3_binary61_predictions50_imagenet_test7600.npy
│   ├── GoogLeNet_depth4_binary61_predictions50_imagenet_test7600.npy
│   ├── GoogLeNet_depth5_binary61_predictions50_imagenet_test7600.npy
│   ├── GoogLeNet_depth6_binary61_predictions50_imagenet_test7600.npy
│   ├── GoogLeNet_depth7_binary61_predictions50_imagenet_test7600.npy
│   ├── GoogLeNet_depth8_binary61_predictions50_imagenet_test7600.npy
│   ├── GoogLeNetSPP_binary61_predictions50_imagenet_test7600.npy
│   ├── GoogLeNetSPP_binary61_predictions50_test7600.npy
│   ├── GoogLeNetSPP_binary61_predictionsfrom50_test7600.npy
│   ├── GoogLeNetSPP_binary61_predictions_imagenet_test7600.npy
│   ├── GoogLeNetSPP_binary61_predictions_test7600.npy
│   ├── GoogLeNetSPP_binary61_predictions_test.npy
│   ├── GoogLeNetSPP_binary61_X_test.npy
│   ├── GoogLeNetSPP_binary61_y_test.npy
│   ├── hiarBayesGoogLeNet_binary3_binary3_epoch50_valloss0.40.h_predictions_imagenet_test_RAP.npy
│   ├── hiarBayesGoogLeNet_binary51_binary51_final_model_predictions_imagenet_test_RAP.npy
│   ├── hiarBayesGoogLeNet_binary61_add_predictions50_imagenet_test7600.npy
│   ├── hiarBayesGoogLeNet_binary61_cond_predictions50_imagenet_test7600.npy
│   ├── hiarBayesGoogLeNet_binary61_gap&dense_binary61_gap&dense_final500iter_model_predictions_imagenet_test7600.npy
│   ├── hiarBayesGoogLeNet_binary61_gap&dense_binary61_gap&dense_final_model_predictions_imagenet_test7600.npy
│   ├── hiarBayesGoogLeNet_binary61_multi_500_predictions_imagenet_test7600.npy
│   ├── hiarBayesGoogLeNet_binary61_multi_loss_predictions50_imagenet_test7600.npy
│   ├── hiarBayesGoogLeNet_binary61_multi_mar_500_predictions_imagenet_test7600.npy
│   ├── hiarBayesGoogLeNet_binary61_multi_mar_predictions50_imagenet_test7600.npy
│   ├── hiarBayesGoogLeNet_binary61_pcm_predictions50_imagenet_test7600.npy
│   ├── hiarBayesGoogLeNet_binary61_predictions50_imagenet_test7600.npy
│   ├── hiarBayesGoogLeNet_binary61_predictions50_test7600.npy
│   ├── hiarBayesGoogLeNet_binary61_predictions_imagenet_test7600.npy
│   ├── hiarBayesGoogLeNet_binary61_predictions_test7600.npy
│   ├── hiarBayesGoogLeNet_binary61_predictions_test.npy
│   ├── hiarBayesGoogLeNet_binary61_rl_predictions50_imagenet_test7600.npy
│   ├── hiarBayesGoogLeNet_binary61_X_test.npy
│   ├── hiarBayesGoogLeNet_binary61_y_test.npy
│   ├── hiarBayesGoogLeNet_binary92_binary92_final_model_predictions_imagenet_test_RAP.npy
│   ├── hiarBayesGoogLeNet_binary92_predictions_imagenet_test_RAP.npy
│   ├── hiarBayesGoogLeNet_binary9_predictions500_imagenet_berk.npy
│   ├── hiarBayesGoogLeNet_binary9_predictions_imagenet_berk.npy
│   ├── hiarGoogLeNet_binary51_binary51_final_model_predictions_imagenet_test_RAP.npy
│   ├── hiarGoogLeNet_binary61_cluster_predictions50_imagenet_test7600.npy
│   ├── hiarGoogLeNet_binary61_predictions50high_test7600.npy
│   ├── hiarGoogLeNet_binary61_predictions50_imagenet_test7600.npy
│   ├── hiarGoogLeNet_binary61_predictions50_test7600.npy
│   ├── hiarGoogLeNet_binary61_predictions_imagenet_test7600.npy
│   ├── hiarGoogLeNet_binary61_predictions_test7600.npy
│   ├── hiarGoogLeNet_binary61_predictions_test.npy
│   ├── hiarGoogLeNet_binary61v2_predictions50_imagenet_test7600.npy
│   ├── hiarGoogLeNet_binary61_X_test.npy
│   ├── hiarGoogLeNet_binary61_y_test.npy
│   ├── hiarGoogLeNet_binary92_binary92_epoch50_valloss0.20.h_predictions_imagenet_test_RAP.npy
│   ├── hiarGoogLeNet_binary92_binary92_final_model_predictions_imagenet_test_RAP.npy
│   ├── hiarGoogLeNet_high_binary61_predictions50_imagenet_test7600.npy
│   ├── hiarGoogLeNet_high_binary61_predictions_imagenet_test7600.npy
│   ├── hiarGoogLeNet_low_binary61_duan_predictions50_imagenet_test7600.npy
│   ├── hiarGoogLeNet_low_binary61_predictions50_imagenet_test7600.npy
│   ├── hiarGoogLeNet_mid_binary61_duan_predictions50_imagenet_test7600.npy
│   ├── hiarGoogLeNet_mid_binary61_predictions50_imagenet_test7600.npy
│   ├── hiarGoogLeNetSPP_binary61_predictions50_imagenet_test7600.npy
│   ├── hiarGoogLeNetSPP_binary61_predictions50_test7600.npy
│   ├── hiarGoogLeNetSPP_binary61_predictions_imagenet_test7600.npy
│   ├── hiarGoogLeNetSPP_binary61_predictions_test7600.npy
│   ├── hiarGoogLeNetSPP_binary61_predictions_test.npy
│   ├── hiarGoogLeNetSPP_binary61_X_test.npy
│   ├── hiarGoogLeNetSPP_binary61_y_test.npy
│   ├── hiarGoogLeNetWAM_binary61_predictions50_imagenet_test7600.npy
│   ├── hiarGoogLeNetWAM_binary61_predictions50_test7600.npy
│   ├── hiarGoogLeNetWAM_binary61_predictions_imagenet_test7600.npy
│   ├── hiarGoogLeNetWAM_binary61_predictions_test7600.npy
│   ├── hiarGoogLeNetWAM_binary61_predictions_test.npy
│   ├── hiarGoogLeNetWAM_binary61_X_test.npy
│   └── hiarGoogLeNetWAM_binary61_y_test.npy
├── RAP_labels_pd.csv
├── RAP_partion.npy
├── RAP_partion.txt
├── relation_array.npy
├── samples
│   ├── RAP_sample_all_model_filter0.jpg
│   ├── RAP_sample_all_model_filter10.jpg
│   ├── RAP_sample_all_model_filter11.jpg
│   ├── RAP_sample_all_model_filter12.jpg
│   ├── RAP_sample_all_model_filter13.jpg
│   ├── RAP_sample_all_model_filter14.jpg
│   ├── RAP_sample_all_model_filter15.jpg
│   ├── RAP_sample_all_model_filter16.jpg
│   ├── RAP_sample_all_model_filter17.jpg
│   ├── RAP_sample_all_model_filter18.jpg
│   ├── RAP_sample_all_model_filter19.jpg
│   ├── RAP_sample_all_model_filter1.jpg
│   ├── RAP_sample_all_model_filter2.jpg
│   ├── RAP_sample_all_model_filter3.jpg
│   ├── RAP_sample_all_model_filter4.jpg
│   ├── RAP_sample_all_model_filter5.jpg
│   ├── RAP_sample_all_model_filter6.jpg
│   ├── RAP_sample_all_model_filter7.jpg
│   ├── RAP_sample_all_model_filter8.jpg
│   ├── RAP_sample_all_model_filter9.jpg
│   ├── RAP_sample_all_model_merge.jpg
│   ├── sample_all_model_filter0.jpg
│   ├── sample_all_model_filter10.jpg
│   ├── sample_all_model_filter11.jpg
│   ├── sample_all_model_filter12.jpg
│   ├── sample_all_model_filter13.jpg
│   ├── sample_all_model_filter14.jpg
│   ├── sample_all_model_filter15.jpg
│   ├── sample_all_model_filter16.jpg
│   ├── sample_all_model_filter17.jpg
│   ├── sample_all_model_filter18.jpg
│   ├── sample_all_model_filter19.jpg
│   ├── sample_all_model_filter1.jpg
│   ├── sample_all_model_filter2.jpg
│   ├── sample_all_model_filter3.jpg
│   ├── sample_all_model_filter4.jpg
│   ├── sample_all_model_filter5.jpg
│   ├── sample_all_model_filter6.jpg
│   ├── sample_all_model_filter7.jpg
│   ├── sample_all_model_filter8.jpg
│   ├── sample_all_model_filter9.jpg
│   ├── sample_gh.jpg
│   ├── sample_g.jpg
│   ├── sample_h.jpg
│   └── sample.jpg
├── state_transition_matrix.npy
├── test_predictions.csv
└── title.csv

```
