import os
import sys
import random
import itertools
import functools
import imageio
import cv2
import numpy as np
import matplotlib.pyplot as plt


def imageToChromosome(imageArray: np.ndarray) -> np.ndarray:
    # Görüntü boyutlarından toplam piksel sayısına göre tek satırlık vektör shape'i oluşturur.
    newShape: np.ndaray = (functools.reduce(
        lambda a, b: a * b, imageArray.shape))

    # Ve bu shape'e göre kromozom oluşturulur.
    chromosome: np.ndarray = np.reshape(a=imageArray, newshape=newShape)
    return chromosome


def chromosomeToImage(chromosome: np.ndarray, imageShape: tuple) -> np.ndarray:
    # Görüntüye ait shape'e göre chromosome'un datası kullanılarak görüntü array'i oluşturulur.
    imageArray: np.ndarray = np.reshape(a=chromosome, newshape=imageShape)
    return imageArray


def initPopulation(imageShape: tuple, numberOfIndividuals: int = 8) -> np.ndarray:
    # Popülasyondaki kromozom / çözüm sayısına göre boş bir popülasyon oluşturulur.
    population: np.ndarray = np.empty(shape=(numberOfIndividuals,
                                             functools.reduce(lambda a, b: a * b, imageShape)),
                                      dtype=np.uint8)

    # Her bir kromozom için [0, 255] arasında rastgele değerler atanır.
    for individualIndex in range(numberOfIndividuals):
        population[individualIndex, :] = np.random.random(
            functools.reduce(lambda a, b: a * b,
                             imageShape)) * 256
    return population


def fitnessFn(targetChromosome: np.ndarray, individualChromosome: np.ndarray) -> float:
    # Her bir çözüm / kromozom için fitness hesaplanır.
    # Buradaki fitness değeri de oluşturulan kromozomun genleri ile görüntüden elde edilen kromozomun
    # genleri arasındaki fark ile elde edilir.
    # -1 ise burada kalite belirteci olarak kullanılmaktadır, negatif değer ile başlayıp 0'a ulaşması beklenir.
    quality: float = -1 * \
        np.sum(np.abs(individualChromosome - targetChromosome))
    return quality


def calculatePopulationFitness(targetChromosome: np.ndarray, population: np.ndarray) -> np.ndarray:
    # Popülasyondaki tüm çözümlerin / kromozomların fitness değerlerini hesaplar.
    qualities = np.zeros(population.shape[0])
    for individualIndex in range(population.shape[0]):
        qualities[individualIndex] = fitnessFn(
            targetChromosome, population[individualIndex, :])
    return qualities


def selectMatingPool(pop: np.ndarray, qualities: np.ndarray, numberOfParents: int) -> np.ndarray:
    # Mevcut jenerasyonda sonraki jenerasyonun daha iyi olması için en iyileri seçerek birbiriyle eşler.
    parents: np.ndarray = np.empty(
        (numberOfParents, pop.shape[1]), dtype=np.uint8)
    for parentNumber in range(numberOfParents):
        # Seçilmemiş en iyiyi seçer.

        # Kalite içerisinde maksimum kalite değerini sağlayan ya da sağlayanlar var ise alır.
        maxQualityArrayContainer: tuple = np.where(
            qualities == np.max(qualities))

        # print(
        #     f"qualities: {qualities=}\nnp.max(qualities): {np.max(qualities)=}\n isEq?: {qualities == np.max(qualities)=}\nnp.where: {np.where(qualities == np.max(qualities))=}")

        # qualities: qualities=array([-1.00000000e+00,  4.56781429e+06,  4.56781428e+06,  4.56781428e+06,
        # 4.56781428e+06, -1.00000000e+00,  4.56781428e+06,  4.56781428e+06])
        # np.max(qualities): np.max(qualities)=4567814.285831286
        # isEq?: qualities == np.max(qualities)=array([False,  True, False, False, False, False, False, False])
        # np.where: np.where(qualities == np.max(qualities))=(array([1]),)

        # qualities: qualities=array([-1.00000000e+00, -1.00000000e+00,  4.56781428e+06,  4.56781428e+06,
        # 4.56781428e+06, -1.00000000e+00,  4.56781428e+06,  4.56781428e+06])
        # np.max(qualities): np.max(qualities)=4567814.280354218
        # isEq?: qualities == np.max(qualities)=array([False, False, False, False, False, False, False,  True])
        # np.where: np.where(qualities == np.max(qualities))=(array([7]),)

        maxQualityIndex: int = maxQualityArrayContainer[0][0]
        parents[parentNumber, :] = pop[maxQualityIndex, :]

        # Seçilenlerin tekrar seçilmemesi için kalite değeri -1'e eşitlenir.
        qualities[maxQualityIndex] = -1
    return parents


def crossover(parents: np.ndarray, imageShape: tuple, numberOfIndividuals: int = 8) -> np.ndarray:
    # Seçilmiş ebeveynlere cross-over uygulanarak yeni bir popülasyon oluşturulur.
    newPopulation: np.ndarray = np.empty(shape=(numberOfIndividuals,
                                                functools.reduce(lambda a, b: a * b, imageShape)),
                                         dtype=np.uint8)

    # Yeni popülasyonun bireylerinin önceki popülasyona göre daha başarısız olma durumuna karşılık
    # Popülasyonun tamamı yeni bireylerle oluşturulmaz, önceki popülasyonun en iyileri yani ebeveynler kullanılır.
    # Böylelikle genel başarı oranı stabil tutulmaya çalışılır.

    newPopulation[0:parents.shape[0], :] = parents

    # Eğer popülasyon genişliği 10 ve eşleşen ebeveyn sayısı 5 ise yeni çocuk sayısı da 5 olur.
    numberOfNewlyGenerated: int = numberOfIndividuals - parents.shape[0]
    # Seçilmiş ebeveynler ile gerçekleşebilecek tüm permütasyonlar hesaplanır.
    parentPermutations: list = list(itertools.permutations(
        iterable=np.arange(0, parents.shape[0]), r=2))
    # Yeni bireyler oluşturmak için ebeveynler rastgele seçilir.
    selectedPermutations = random.sample(range(len(parentPermutations)),
                                         numberOfNewlyGenerated)

    combinationIndex: int = parents.shape[0]
    for combination in range(len(selectedPermutations)):
        # Yeni bireyler oluşturulur.
        selectedCombinationIndex: int = selectedPermutations[combination]
        selectedCombination: tuple = parentPermutations[selectedCombinationIndex]

        # İki ebeveyn arasında genlerin yarısı birbiriyle değiştirilir.
        # Kesim noktası (cutpoint), toplam gen sayısının yarısı kadardır.
        cutPoint = np.int32(newPopulation.shape[1] / 2)
        newPopulation[combinationIndex + combination, 0:cutPoint] = parents[selectedCombination[0],
                                                                            0:cutPoint]
        newPopulation[combinationIndex + combination, cutPoint:] = parents[selectedCombination[1],
                                                                           cutPoint:]
    return newPopulation


def mutation(population: np.ndarray, numberOfParentsMating: int, mutationPercent: float) -> np.ndarray:
    # Genler belirlenmiş oran ile rastgele seçilir ve rastgele seçilen genler kendi aralarında rastgele değiştirilir.
    for index in range(numberOfParentsMating, population.shape[0]):
        mutationRate: float = mutationPercent / 100 * \
            population.shape[1]

        # Genler mutasyon oranına göre rastgele seçilir.
        rankIndex: int = np.uint32(np.random.random(size=np.uint32(mutationRate))
                                   * population.shape[1])

        # Rastgele seçilen genleri [0, 255] için rastgele değiştirir.
        newValues: np.ndarray = np.uint8(
            np.random.random(size=rankIndex.shape[0]) * 256)

        # Mutasyon sonrası popülasyonu günceller.
        population[index, rankIndex] = newValues
    return population


def displayImage(currentIteration: int, qualities: np.ndarray, newPopulation: np.ndarray, imageShape: tuple,
                 displayPoint: int = 5000):
    # Gösterim için belirlenen iterasyon sayısına göre elde edilen görüntünün anlık durumu ekrana bastırılır.
    if (np.mod(currentIteration, displayPoint) == 0):
        # Jenerasyondaki en başarılı birey seçilir.
        bestSolutionChromosome: np.ndarray = newPopulation[np.where(
            qualities == np.max(qualities))[0][0], :]

        # Bireyin kromozomu görüntü array'ine dönüştürülür.
        bestSolutionImage: np.ndarray = chromosomeToImage(
            bestSolutionChromosome, imageShape)

        plt.imshow(bestSolutionImage)
        plt.show()

        # plt.imshow(bestSolutionImage)
        # plt.show(block=False)
        # plt.pause(0.15)
        # plt.close()


def saveImage(currentIteration: int, qualities: np.ndarray, newPopulation: np.ndarray, imageShape: tuple,
              savePoint: int, saveDirectory: str) -> None:
    # print("{currentIteration=}")
    # Kaydetme için belirlenen iterasyon sayısına göre elde edilen görüntünün anlık durumu dosya olarak kaydedilir.
    if (np.mod(currentIteration, savePoint) == 0):
        # Jenerasyondaki en başarılı birey seçilir.
        bestSolutionChromosome: np.ndarray = newPopulation[np.where(qualities ==
                                                                np.max(qualities))[0][0], :]
        # Bireyin kromozomu görüntü array'ine dönüştürülür.
        bestSolutionImage: np.ndarray = chromosomeToImage(
            bestSolutionChromosome, imageShape)

        # Dönüştürülen görüntü dosyaya kaydedilir.
        plt.imsave(saveDirectory + "solution_" + str(currentIteration) +
                   ".png", bestSolutionImage)


def showIndividuals(individuals: np.ndarray, imageShape: tuple) -> None:
    # Tüm bireyleri gösterir.

    numberOfIndividuals = individuals.shape[0]
    figureRow = 1
    figureColumn = 1
    for k in range(1, np.uint16(individuals.shape[0]/2)):
        if np.floor(np.power(k, 2) / numberOfIndividuals) == 1:
            figureRow = k
            figureColumn = k
            break
    figure, axis = plt.subplots(figureRow, figureColumn)

    currentIndex = 0
    for rowIndex in range(figureRow):
        for columnIndex in range(figureColumn):
            if(currentIndex >= individuals.shape[0]):
                break
            else:
                currentImage = chromosomeToImage(
                    individuals[currentIndex, :], imageShape)
                axis[rowIndex, columnIndex].imshow(currentImage)
                currentIndex = currentIndex + 1
    plt.show()


def main() -> None:
    # Görüntü dosyasını okur.
    targetImage = imageio.imread("images/elon-lol2.jpg")
    # Görüntü dosyasının shape'ini alır.
    height, width, channels = targetImage.shape

    # Daha hızlı iterasyon testi için resim boyutu azaltılır.
    heightMultiplier = 1 / 1
    widthMultiplier = 1 / 1
    height = int(height * heightMultiplier)
    width = int(width * widthMultiplier)
    dimensions = (width, height)
    targetImage = cv2.resize(targetImage, dimensions)

    # Görüntü dosyası kromozoma dönüştürülür.
    targetChromosome = imageToChromosome(targetImage)

    # Popülasyon Genişliği: Popülasyonun sahip olabileceği maksimum çözüm, birey ya da kromozom sayısı.
    solutionPerPopulation = 8
    # Eşleşme Havuzu Genişliği: Eşleşebilecek maksimum çözüm, birey ya da kromozom sayısı.
    numberOfParentsMating = 4
    # Mutasyon Yüzdesi
    mutationPercent = .01
    # İterasyon Sayısı
    iterationCount = 10000001
    # Kaydetme Noktası
    savePoint = 5000
    # Görüntüleme Noktası
    displayPoint = 5000

    # Bazı durumlarda eşleşen ebeveyn sayısı, yeni bir jenerasyon üretmek / popülasyon oluşturmak için yeterli olmayabilir.
    # Bu durumun kontrollü bir hataya sebep olabilmesi için permütasyon ile mümkünlük kontrol yapılır.
    numberOfPossiblePermutations = len(list(itertools.permutations(iterable=np.arange(0,
                                                                                      numberOfParentsMating), r=2)))
    numberOfRequiredPermutations = solutionPerPopulation - numberOfPossiblePermutations
    if (numberOfRequiredPermutations > numberOfPossiblePermutations):
        print("Sağlanan popülasyon genişliği veya ebeveyn sayısı ile eşleşme havuzu genişliği uyumsuzluğu nedeniyle program durduruldu.")
        sys.exit(1)

    # Yeni popülasyon oluşturur.
    newPopulation = initPopulation(imageShape=targetImage.shape,
                                   numberOfIndividuals=solutionPerPopulation)

    for iteration in range(iterationCount):
        # Popülasyondaki her bir kromozom için fitness hesaplanır.
        qualities = calculatePopulationFitness(
            targetChromosome, newPopulation)
        print(
            f"Quality: {np.max(qualities)}, Iteration: {iteration}, {qualities=}")

        # Eşleşme için en iyi ebeveynler seçilir.
        parents = selectMatingPool(newPopulation, qualities,
                                   numberOfParentsMating)

        # Cross-over ile yeni jenerasyon oluşturulur.
        newPopulation = crossover(parents, targetImage.shape,
                                  numberOfIndividuals=solutionPerPopulation)

        # Mutasyon oranının yükseltilmesi ileriki jenerasyonlarda başarısızlık oranını artırabilir.
        # Bu yüzden oran olabildiğince düşük tutulmaktadır.
        newPopulation = mutation(population=newPopulation,
                                 numberOfParentsMating=numberOfParentsMating,
                                 mutationPercent=mutationPercent)

        # Güncel iterasyon sayısındaki en iyi bireyi göster.
        # displayImage(iteration, qualities, newPopulation, targetImage.shape,
        #              displayPoint=displayPoint)
        # Güncel iterasyon sayısındaki en iyi bireyi kaydet.
        saveImage(iteration, qualities, newPopulation, targetImage.shape,
                  savePoint=savePoint, saveDirectory=os.curdir + "/points/")

    # Son jenerasyonu göster.
    showIndividuals(newPopulation, targetImage.shape)


if __name__ == "__main__":
    try:
        os.mkdir(os.curdir + "/points")
    except:
        pass
    main()
