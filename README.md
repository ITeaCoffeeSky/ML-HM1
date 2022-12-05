
HW1_Regression_with_inference.ipynb - ноутбук со всеми проведёнными вами экспериментами (output'ы ячеек, разумеется, сохранил)

VAR1_main.py, VAR2_main - файлы с реализацией сервиса
	VAR1_main - без pickle-файла
	Скрины работы сервиса VAR1 - скрины
	
	VAR2_main - с pickle-файлом
	Скрины работы сервиса VAR2 - скрины
	- predict_items вылетает с ошибкой в сервисе (в ноутбуке код работает), не нашел пока, почему - update: починил, теперь работает

model.pickle - файл с сохранёнными весами модели, коэффициентами скейлинга и прочими числовыми значениями, которые могут понадобиться для инференса (model.joblib - еще и в этом формате)

data.txt - текстовый файл с данными для реализации сервиса

README.md - файл с выводами про проделанной вами работе:
что было сделано
	- выполнено ДЗ по ноутбуку, в нем подробные комментарии
	- EDA: сделал, что требовалось. Много времени ушло на простые, казалось бы, действия с данными датафрейма (парсинг по строкам с реплейсом, дроп с условием, формат вывода чисел и т.д.). зато много нового узнал. Как вариант, надо было попробовать не заполнять пропуски ничем, удалить совсем (всего 200 строк, 3,5% от всех). Еще надо бы было распарсить название на марку и модель, уверен там есть корреляция с целевой переменной. Еще надо бы было попробовать выбросы (из 0,99 квантиля или типа того) заменить на последние максимальные/минимальные, возможно получилось бы повысить качество данных и итоговой модели.
	- Визуализация: сделал как предлагалось (pairplot и heatmap годнаые штуки). Сделал предложения по доп визуализации: совмещение трейна и теста на pairplot (так нагляднее), построил график сравнения убывания стоимости авто с возрастом, отобразил экстремальные значения.
	- Вещественные признаки: сделал по ноутбуку. Увидел как самые коррелирующие с таргетом (из heatmap) признаки получили бОльшие веса. Получил только весьма посредственные R2, подозрительно большой MSE (грешу на выбросы). Причем, обе метрики схожи и на трейн/тест и на разных линейных моделях. Увидел как L1 обнулила несколько весов, качество это не бустануло конечно, но на данных наблюдать, то что на теории было - крайне важно. Сделал подбор ГридСерчем с фолдами, опять же без особых прорывов в качестве предсказания.
	- Категориальные фичи: использовал onehot (из sklearn и из pandas). Sklearn показался громоздким (ну или у меня так получилось), но более функциональным (больше параметров функции). Далее работал с результатами onehot из pandas. Сделал датафрейм с вещественными и onehot категориальными признаками, прошелся ГридСерчем - получил увелечение R2 (ощутимо, но не радикально). В итоге, в сервисе решил использовать лучшую модель по метрики - линейную регрессию. По результату, также остался 1 вопрос: после onehot на трейне и обучения некой модели на этих данных - как использовать в этой модели тест данные (в тесте другие значения категориальных фич, может не быть схождения с моделью, так и получилось)? Как это правильно делать - пока не выяснил. Возможно, просто надо склеить трейн и тест, потом onehot, и потом кросс-валидация - может так оптимально.
	- Feature Eng: не успел, будет время попробую бизнес метрику хотябы до 0,3 дотянуть (основные надежды возлагаю на выкидывание пустых строк, выбросов и парсинг названия)
	- Бизнесовая часть: написал функцию, протестил на трейне с избранной моделью, взгрустнул - 21,3% не поражают воображения даже неприхотливого заказчика.
	- FastAPI сервис: сделал в 2х вариантах. 1й вариант - в лоб, сохранил веса и прочие данные модели в файлик, заложил их в код программы, заработало. Недостаток этого варианта: все в коде, в таком виде модель неудобно менять без перезапуска сервиса, что не правильно концептуально. 2й вариант - с pickle файлом (с весами модели) (однако, автоматизацию EDA данных из запроса в сервисе сделал также в коде, колхозненько, надо бы что-то универсальнее и изящнее). Сервис крайне примитивный (см скрины), не успел сделать UI, загрузку  csv, проверку хоть каких-то ошибок. Протестил через docs - работает! (мелочь, а приятно)
	- офрмление результатов: был не внимателен, много написал в ноутбук, мало в md. Вот исправляюсь. Надеюсь, это не будет расценено как пропуск дедлайна, основные результаты ДЗ не менял с момента сдачи.
	
с какими результатами
	- результаты здесь. Что можно отметить: 
		качество модели низко
		бизнес-метрика также не радует
		подробно описал в предыдущем пункте
		
что дало наибольший буст в качестве
	- добавление категориальных фич к вещественным. Лучший результат показала линейная регрессия. Уверен при помощи FE можно заметно повысить качество предсказаний.
	
что сделать не вышло и почему (это нормально, даже хорошо😀)
	- много чего не успел (поброности выше):
		- попробовать больше вариантов увеличения качества и бизнес-метрики
		- Feature Eng (были идеи, см "что было сделано")
		- разделение и использование столбца torque
		- починить сервис, 2 вариант (сделал позже)
		- сделать UI и загрузку файлов для сервиса
