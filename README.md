## Dosya İçeriği

1. *** jupyter notebooks klasörü ***
    -Bu klasörde projeye baslamadan once veri setiyle ilgili yapılan calısmalar var.
    sadece streamlit entegrasyonu yapmadan once tamamen calısır bir kod elde edebilmek
    adına yapılan calısmalar ve cesitli veri gorselleştirmeleri mevcut.

2. *** modules klasörü *** 
    -Streamlit kaynak dosyasında calıstıracagımız ihtiyacımız olan modelleri ve içeriklerini barındıran python modulu.

3. *** M5_streamlit_imp.py dosyası ***
    -Streamlit implementasyonunun gerceklestigi dosya. bu dosyadan projemizi ayağa kaldıracağız

## NOTES
1. Projede yapılan preprocess işlemleri bize verilen jupyter notebooks klasörü icindeki data.csv ' ye gore yapıldı . Eger baska yüklenen veride bir sıkıntı cıkarsa preprocess den dolayı(ornegin atılan raw data da bir virgül daha var sonda ve unnamed 32 oluyor preprocessle bunu kaldırdık fakaat baska atılan datanın son sütununda unnamed : 32 yer almazsa sıkıntı hata alınabilir. Gerci uygun kontrol yapılarını koydum ama güvenemiyorum.) lütfen jupyter notebookdaki data ' yı yükleyerek projeyi gorun.

2. Projede sizlere zorluk cıkarmaması acısından okuması da kolay olsun diye ekstra bir kütüphane kullanmadım. Derste gordugumuz numpy , pandas , seaborn , sklearn gibi kütüphaneler var.

3. Cesitli konseptleri jupyter notebook da denedim fakat denedigim her seyi implemente etmedim bakabilirsiniz.

4. Tekrardan her sey icin emekleriniz icin sonsuz tesekkürlerimi sunuyorum.
## Nasıl Çalıştırılır

1. GEREKLİ PAKETLERİ YUKLEDİKTEN SONRA KOMUT SATIRINA ASAGIDAKİ KODU YAPISTIRIN:

```bash
streamlit run M5_streamlit_imp.py