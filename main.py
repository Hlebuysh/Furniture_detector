from keras.models import load_model
import keras.utils as image
from keras.applications.vgg16 import preprocess_input
import numpy as np

# Загрузка обученной модели
model = load_model(r'C:\Users\Hlebush\Downloads\furniture_detector.h5')  # Укажите путь к вашей сохраненной модели

# Список названий категорий самой модели
class_list = ['bed', 'chair', 'sofa', 'swivelchair', 'table']

# Размер изображения, который ожидает модель
img_height, img_width = 224, 224


# Подготовка изображения для передачи в модель
def preprocess_image(img_path):
    # Загрузка изображения и изменение его размера до требуемых размеров
    img = image.load_img(img_path, target_size=(img_height, img_width))

    # Преобразование изображение в массив numpy, чтобы его можно было обрабатывать программно.
    img_array = image.img_to_array(img)

    # Добавляет одно измерение в начало массива для создания "батча". Модель ожидает батч изображений.
    img_array = np.expand_dims(img_array, axis=0)

    # Преобразование изображение в формат, ожидаемый моделью
    img_array = preprocess_input(img_array)
    return img_array


# Ввод пути к изображению из консоли
image_path = input("Введите путь к изображению: ")

# Проверка корректности пути к изображению
try:
    img_array = preprocess_image(image_path)
except Exception as e:
    print("Ошибка:", e)
    exit()

# Предсказание категории

# Метод predict принимает массив изображений, в данном случае, это предобработанный массив с изображением,
# и возвращает предсказанные вероятности для каждого класса.
predictions = model.predict(img_array)

# Функция argmax из нахождит индекс класса с наибольшей предсказанной вероятностью.
predicted_class_index = np.argmax(predictions)

# Извлечение имени наиболее вероятного класса
predicted_class = class_list[predicted_class_index]

# Вывод результата предсказания
print(f"Наиболее вероятная категория: {predicted_class}")
