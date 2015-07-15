#     with open(settings.LABELS_PATH, 'w') as label_file:
#         max_w = 0
#         max_h = 0
#         for i, img in enumerate(img_names):
# #            print 'Processing image: '+str(i)+'\n'
#             img_path = settings.INHABITED_PATH+img
#             xml_path = settings.INHABITED_PATH+img[:-3]+'xml'

#             roof_list, cur_max_w, cur_max_h = loader.get_roofs(xml_path)
#            # max_h = cur_max_h if (max_h<cur_max_h) else max_h
#            # max_w = cur_max_w if (max_w<cur_max_h) else max_h

#             for r, roof in enumerate(roof_list):
# #                print 'Processing roof: '+str(r)+'\n'
#                 loader.produce_roof_patches(img_path=img_path, img_id=i+1, 
#                                     roof=roof, label_file=label_file, max_h=max_h, max_w=max_w)
#         neg_patches_wanted = settings.NEGATIVE_PATCHES_NUM*loader.total_patch_no
#         loader.get_negative_patches(neg_patches_wanted, label_file)
#         #settings.print_debug('************* Total patches saved: *****************: '+str(loader.total_patch_no))

