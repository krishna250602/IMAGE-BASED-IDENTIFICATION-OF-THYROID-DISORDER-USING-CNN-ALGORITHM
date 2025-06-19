from simple_image_download import simple_image_download as sim

my_downloader = sim.Downloader()

my_downloader.directory = 'data_set/'
# Change File extension type
my_downloader.extensions = '.jpg'
print(my_downloader.extensions)
my_downloader.download('Hyperthyroidism_ultrasoung_images', limit=48)
# my_downloader.download('thyroid_cancer_ultrasound_images', limit=105)
