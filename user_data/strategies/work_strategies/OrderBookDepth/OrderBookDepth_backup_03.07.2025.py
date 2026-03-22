from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from pandas import DataFrame
import pandas as pd
import numpy as np
from typing import Dict, List
from freqtrade.exchange.exchange_types import OrderBook
import logging

logger = logging.getLogger(__name__)


class OrderBookDepthStrategy(IStrategy):
    """
    Стратегия на основе анализа глубины стакана.

    Основные принципы работы:
    1. Анализ глубины стакана для поиска кластеров ликвидности
    2. Размещение ордеров рядом с этими кластерами
    3. Использование анализа объема для подтверждения силы уровней
    """

    # Параметры стратегии
    timeframe = "5m"  # Таймфрейм для анализа

    # Параметры анализа стакана
    order_book_depth = IntParameter(
        20, 100, default=50, space="buy", description="Глубина анализа стакана"
    )
    min_cluster_size = DecimalParameter(
        0.1, 1.0, default=0.1, space="buy", description="Минимальный размер кластера ликвидности"
    )
    cluster_price_threshold = DecimalParameter(
        0.0001,
        0.001,
        default=0.0005,
        space="buy",
        description="Порог цены для объединения кластеров (относительное изменение цены)",
    )
    min_orders_in_cluster = IntParameter(
        2, 10, default=3, space="buy", description="Минимальное количество ордеров в кластере"
    )

    # Параметры анализа объема
    volume_threshold = DecimalParameter(
        1.0, 5.0, default=2.0, space="buy", description="Порог объема для подтверждения уровня"
    )
    min_volume_ratio = DecimalParameter(
        0.5,
        2.0,
        default=1.0,
        space="buy",
        description="Минимальное соотношение объемов покупки/продажи",
    )

    # Параметры управления рисками
    stoploss = -0.02  # Стоплосс 2%
    minimal_roi = {
        "0": 0.05,  # 5% прибыли сразу
        "30": 0.025,  # 2.5% после 30 минут
        "60": 0.015,  # 1.5% после 1 часа
        "120": 0.01,  # 1% после 2 часов
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Добавление индикаторов и данных о кластерах ликвидности
        """
        try:
            pair = metadata.get("pair", "unknown")

            # Получаем данные стакана с указанной глубиной
            orderbook = self.dp.exchange.fetch_l2_order_book(pair, self.order_book_depth.value)

            # Анализируем кластеры ликвидности
            buy_clusters = self._find_liquidity_clusters(orderbook["bids"], "bids")
            sell_clusters = self._find_liquidity_clusters(orderbook["asks"], "asks")

            # Берем топ-2 кластера с каждой стороны
            top_buy_clusters = buy_clusters[:2] if buy_clusters else [self._get_empty_cluster()]
            top_sell_clusters = sell_clusters[:2] if sell_clusters else [self._get_empty_cluster()]

            # Добавляем информацию о кластерах в dataframe
            # Основные кластеры (самые большие по объему)
            dataframe["buy_cluster_price"] = top_buy_clusters[0].get("price", 0)
            dataframe["buy_cluster_volume"] = top_buy_clusters[0].get("total_volume", 0)
            dataframe["buy_cluster_orders"] = top_buy_clusters[0].get("order_count", 0)

            dataframe["sell_cluster_price"] = top_sell_clusters[0].get("price", 0)
            dataframe["sell_cluster_volume"] = top_sell_clusters[0].get("total_volume", 0)
            dataframe["sell_cluster_orders"] = top_sell_clusters[0].get("order_count", 0)

            # Вторые по величине кластеры (если есть)
            if len(top_buy_clusters) > 1:
                dataframe["buy_cluster2_price"] = top_buy_clusters[1].get("price", 0)
                dataframe["buy_cluster2_volume"] = top_buy_clusters[1].get("total_volume", 0)
            else:
                dataframe["buy_cluster2_price"] = 0
                dataframe["buy_cluster2_volume"] = 0

            if len(top_sell_clusters) > 1:
                dataframe["sell_cluster2_price"] = top_sell_clusters[1].get("price", 0)
                dataframe["sell_cluster2_volume"] = top_sell_clusters[1].get("total_volume", 0)
            else:
                dataframe["sell_cluster2_price"] = 0
                dataframe["sell_cluster2_volume"] = 0

            # Рассчитываем дополнительные метрики
            dataframe["volume_ratio"] = dataframe["volume"] / dataframe["volume"].rolling(20).mean()

            # Разница между первым и вторым кластером
            if len(top_buy_clusters) > 1:
                dataframe["buy_cluster_diff"] = abs(
                    top_buy_clusters[0].get("price", 0) - top_buy_clusters[1].get("price", 0)
                ) / (top_buy_clusters[0].get("price", 0) or 1)  # Избегаем деления на ноль
            else:
                dataframe["buy_cluster_diff"] = 0

            if len(top_sell_clusters) > 1:
                dataframe["sell_cluster_diff"] = abs(
                    top_sell_clusters[0].get("price", 0) - top_sell_clusters[1].get("price", 0)
                ) / (top_sell_clusters[0].get("price", 0) or 1)  # Избегаем деления на ноль
            else:
                dataframe["sell_cluster_diff"] = 0

            # Логируем информацию о кластерах
            if not dataframe.empty:
                last_row = dataframe.iloc[-1]
                logger.debug(
                    f"Обновлены данные кластеров для {pair}: "
                    f"BID: {last_row['buy_cluster_price']:.8f} ({last_row['buy_cluster_volume']:.4f}), "
                    f"ASK: {last_row['sell_cluster_price']:.8f} ({last_row['sell_cluster_volume']:.4f})"
                )

            return dataframe

        except Exception as e:
            logger.error(f"Ошибка в populate_indicators для {pair}: {str(e)}", exc_info=True)
            # Возвращаем dataframe с нулевыми значениями в случае ошибки
            default_columns = {
                "buy_cluster_price": 0,
                "buy_cluster_volume": 0,
                "buy_cluster_orders": 0,
                "sell_cluster_price": 0,
                "sell_cluster_volume": 0,
                "sell_cluster_orders": 0,
                "buy_cluster2_price": 0,
                "buy_cluster2_volume": 0,
                "sell_cluster2_price": 0,
                "sell_cluster2_volume": 0,
                "volume_ratio": 0,
                "buy_cluster_diff": 0,
                "sell_cluster_diff": 0,
            }

            for col, default_val in default_columns.items():
                if col not in dataframe.columns:
                    dataframe[col] = default_val

            return dataframe

    def _get_empty_cluster(self) -> Dict:
        """Возвращает пустой кластер с значениями по умолчанию"""
        return {
            "price": 0.0,
            "total_volume": 0.0,
            "order_count": 0,
            "min_price": 0.0,
            "max_price": 0.0,
            "price_range": 0.0,
            "avg_order_volume": 0.0,
        }

    def _find_liquidity_clusters(self, orders: List[tuple], side: str = "bids") -> List[Dict]:
        """
        Поиск кластеров ликвидности в стакане.

        Args:
            orders: Список ордеров (цена, объем)
            side: Сторона стакана ('bids' для покупок, 'asks' для продаж)

        Returns:
            Список словарей с информацией о найденных кластерах, отсортированный по объему (по убыванию)
        """
        if not orders:
            logger.debug(f"Нет данных для анализа кластеров (side={side})")
            return []

        try:
            # Преобразуем данные и фильтруем нулевые объемы
            filtered_orders = []
            for order in orders:
                try:
                    price = float(order[0])
                    volume = float(order[1])
                    if volume > 0 and price > 0:
                        filtered_orders.append((price, volume))
                except (ValueError, IndexError, TypeError) as e:
                    logger.warning(f"Ошибка обработки ордера {order}: {e}")

            if not filtered_orders:
                logger.debug(f"Нет валидных ордеров для анализа (side={side})")
                return []

            # Сортируем ордера в зависимости от стороны стакана
            # Для bids: сортируем по убыванию цены (самые высокие цены первые)
            # Для asks: сортируем по возрастанию цены (самые низкие цены первые)
            filtered_orders.sort(key=lambda x: x[0], reverse=(side == "bids"))
            logger.debug(
                f"Первый ордер после сортировки ({side}): цена={filtered_orders[0][0]}, объем={filtered_orders[0][1]}"
            )

            # Логируем параметры кластеризации
            logger.debug(f"Анализ {side} с {len(filtered_orders)} ордерами")
            logger.debug(
                f"Параметры: cluster_price_threshold={self.cluster_price_threshold.value}, "
                f"min_cluster_size={self.min_cluster_size.value}, "
                f"min_orders_in_cluster={self.min_orders_in_cluster.value}"
            )

            # Логируем первые 10 ордеров после сортировки
            logger.debug(f"\n=== СТАКАН {side.upper()} ПОСЛЕ СОРТИРОВКИ ===")
            logger.debug(f"{'Цена':>15} | {'Объем':>15} | Относительная разница")
            logger.debug("-" * 50)

            for i, (price, volume) in enumerate(
                filtered_orders[:10]
            ):  # Показываем первые 10 ордеров
                if i == 0:
                    price_diff_pct = 0.0
                else:
                    prev_price = filtered_orders[i - 1][0]
                    price_diff = abs(price - prev_price)
                    price_diff_pct = (price_diff / prev_price) * 100

                logger.debug(f"{price:>15.8f} | {volume:>15.8f} | {price_diff_pct:>6.4f}%")

            if len(filtered_orders) > 10:
                logger.debug(f"... и еще {len(filtered_orders) - 10} ордеров")
            logger.debug("=" * 50 + "\n")

            clusters = []
            current_cluster = None

            for price, volume in filtered_orders:
                if current_cluster is None:
                    # Инициализируем первый кластер
                    current_cluster = {
                        "prices": [price],
                        "volumes": [volume],
                        "total_volume": volume,
                        "weighted_sum": price * volume,
                        "min_price": price,
                        "max_price": price,
                    }
                    continue

                # Проверяем, относится ли ордер к текущему кластеру
                last_price = current_cluster["prices"][-1]
                price_diff = abs(price - last_price)
                price_ratio = price_diff / last_price if last_price > 0 else float("inf")

                if price_ratio <= self.cluster_price_threshold.value:
                    # Добавляем ордер в текущий кластер
                    current_cluster["prices"].append(price)
                    current_cluster["volumes"].append(volume)
                    current_cluster["total_volume"] += volume
                    current_cluster["weighted_sum"] += price * volume
                    current_cluster["min_price"] = min(current_cluster["min_price"], price)
                    current_cluster["max_price"] = max(current_cluster["max_price"], price)
                else:
                    # Сохраняем текущий кластер, если он соответствует критериям
                    if (
                        current_cluster["total_volume"] >= self.min_cluster_size.value
                        and len(current_cluster["prices"]) >= self.min_orders_in_cluster.value
                    ):
                        cluster_data = self._create_cluster_data(current_cluster)
                        clusters.append(cluster_data)

                    # Начинаем новый кластер
                    current_cluster = {
                        "prices": [price],
                        "volumes": [volume],
                        "total_volume": volume,
                        "weighted_sum": price * volume,
                        "min_price": price,
                        "max_price": price,
                    }

            # Добавляем последний кластер, если он соответствует критериям
            if (
                current_cluster
                and current_cluster["total_volume"] >= self.min_cluster_size.value
                and len(current_cluster["prices"]) >= self.min_orders_in_cluster.value
            ):
                cluster_data = self._create_cluster_data(current_cluster)
                clusters.append(cluster_data)

            # Сортируем кластеры по объему (по убыванию)
            clusters.sort(key=lambda x: x["total_volume"], reverse=True)

            # Логируем информацию о найденных кластерах
            logger.debug(f"Found {len(clusters)} clusters")
            for i, cluster in enumerate(clusters):
                logger.debug(
                    f"Cluster {i + 1}: price={cluster['price']}, volume={cluster['total_volume']}, count={cluster['order_count']}"
                )

            return clusters

        except Exception as e:
            logger.error(f"Ошибка при поиске кластеров (side={side}): {str(e)}", exc_info=True)
            return []

    def _create_cluster_data(self, cluster: Dict) -> Dict:
        """Создает словарь с данными кластера"""
        if not cluster or not cluster.get("prices") or not cluster.get("volumes"):
            return {}

        total_volume = cluster["total_volume"]
        weighted_price = cluster["weighted_sum"] / total_volume if total_volume > 0 else 0

        return {
            "price": weighted_price,
            "total_volume": total_volume,
            "order_count": len(cluster["prices"]),
            "min_price": cluster["min_price"],
            "max_price": cluster["max_price"],
            "price_range": cluster["max_price"] - cluster["min_price"],
            "avg_order_volume": total_volume / len(cluster["volumes"]) if cluster["volumes"] else 0,
        }

    def _log_clusters(self, clusters: List[Dict], side: str) -> None:
        """
        Логирует информацию о найденных кластерах

        Args:
            clusters: Список кластеров для логирования
            side: Сторона стакана ('bids' или 'asks')
        """
        if not clusters:
            logger.debug(f"Не найдено значимых кластеров (side={side})")
            return

        logger.debug(f"\n=== НАЙДЕНЫ КЛАСТЕРЫ ЛИКВИДНОСТИ ({side.upper()}) ===")
        logger.debug(f"Всего кластеров: {len(clusters)}")

        for i, cluster in enumerate(clusters[:10]):  # Логируем топ-10 кластеров
            price = cluster.get("price", 0)
            total_volume = cluster.get("total_volume", 0)
            order_count = cluster.get("order_count", 0)
            price_range = cluster.get("price_range", 0)
            avg_volume = cluster.get("avg_order_volume", 0)
            min_price = cluster.get("min_price", price)
            max_price = cluster.get("max_price", price)

            logger.debug(
                f"Кластер #{i + 1}:\n"
                f"  Цена: {price:.8f}\n"
                f"  Общий объем: {total_volume:.8f}\n"
                f"  Количество ордеров: {order_count}\n"
                f"  Диапазон цен: {price_range:.8f} ({min_price:.8f} - {max_price:.8f})\n"
                f"  Средний объем ордера: {avg_volume:.8f}\n"
                f"  Плотность: {(total_volume / price_range) if price_range > 0 else float('inf'):.2f} (объем/диапазон)"
            )

            # Дополнительная информация для отладки
            if (
                i == 0 and order_count > 0 and order_count <= 5
            ):  # Для первого кластера с малым количеством ордеров
                prices = cluster.get("prices", [])
                volumes = cluster.get("volumes", [])
                logger.debug("  Детали ордеров в кластере:")
                for j, (p, v) in enumerate(zip(prices, volumes)):
                    logger.debug(f"    Ордер {j + 1}: Цена={p:.8f}, Объем={v:.8f}")

        # Суммарная статистика
        if clusters:
            total_volume = sum(c.get("total_volume", 0) for c in clusters)
            total_orders = sum(c.get("order_count", 0) for c in clusters)
            logger.debug(
                f"\nСуммарно: {len(clusters)} кластеров, "
                f"{total_orders} ордеров, "
                f"общий объем: {total_volume:.8f}"
            )

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Генерация сигналов на вход в позицию на основе анализа кластеров ликвидности
        """
        try:
            pair = metadata.get("pair", "unknown")

            # Инициализируем колонки для сигналов
            dataframe["enter_long"] = 0
            dataframe["enter_short"] = 0

            # Проверяем наличие необходимых колонок
            required_columns = [
                "buy_cluster_volume",
                "sell_cluster_volume",
                "volume_ratio",
                "buy_cluster_price",
                "sell_cluster_price",
                "close",
                "buy_cluster_orders",
                "sell_cluster_orders",
            ]

            if not all(col in dataframe.columns for col in required_columns):
                logger.warning(
                    f"Отсутствуют необходимые колонки для анализа кластеров в паре {pair}"
                )
                return dataframe

            # Получаем последнюю строку с данными
            last_row = dataframe.iloc[-1]

            # Рассчитываем дополнительные метрики для анализа
            buy_pressure = last_row["buy_cluster_volume"] / (last_row["sell_cluster_volume"] or 1)
            sell_pressure = last_row["sell_cluster_volume"] / (last_row["buy_cluster_volume"] or 1)

            # Условия для входа в длинную позицию
            long_conditions = [
                # Объем кластера покупок выше порога
                last_row["buy_cluster_volume"] > self.volume_threshold.value,
                # Отношение текущего объема к среднему выше порога
                last_row["volume_ratio"] > self.min_volume_ratio.value,
                # Цена закрытия выше цены кластера покупок
                last_row["close"] > last_row["buy_cluster_price"],
                # Давление покупателей значительно выше продавцов
                buy_pressure > 1.5,
                # Минимальное количество ордеров в кластере
                last_row["buy_cluster_orders"] >= 3,
            ]

            # Условия для входа в короткую позицию
            short_conditions = [
                # Объем кластера продаж выше порога
                last_row["sell_cluster_volume"] > self.volume_threshold.value,
                # Отношение текущего объема к среднему выше порога
                last_row["volume_ratio"] > self.min_volume_ratio.value,
                # Цена закрытия ниже цены кластера продаж
                last_row["close"] < last_row["sell_cluster_price"],
                # Давление продавцов значительно выше покупателей
                sell_pressure > 1.5,
                # Минимальное количество ордеров в кластере
                last_row["sell_cluster_orders"] >= 3,
            ]

            # Применяем условия к последней свече
            if all(long_conditions):
                dataframe.loc[dataframe.index[-1], "enter_long"] = 1
                logger.info(
                    f"СИГНАЛ НА ПОКУПКУ {pair} | "
                    f"Цена: {last_row['close']:.8f} | "
                    f"Кластер покупок: {last_row['buy_cluster_price']:.8f} (объем: {last_row['buy_cluster_volume']:.4f}, ордеров: {last_row['buy_cluster_orders']}) | "
                    f"Объем/среднее: {last_row['volume_ratio']:.2f}x | "
                    f"Давление покупателей: {buy_pressure:.2f}x"
                )

            if all(short_conditions):
                dataframe.loc[dataframe.index[-1], "enter_short"] = 1
                logger.info(
                    f"СИГНАЛ НА ПРОДАЖУ {pair} | "
                    f"Цена: {last_row['close']:.8f} | "
                    f"Кластер продаж: {last_row['sell_cluster_price']:.8f} (объем: {last_row['sell_cluster_volume']:.4f}, ордеров: {last_row['sell_cluster_orders']}) | "
                    f"Объем/среднее: {last_row['volume_ratio']:.2f}x | "
                    f"Давление продавцов: {sell_pressure:.2f}x"
                )

            return dataframe

        except Exception as e:
            logger.error(f"Ошибка в populate_entry_trend для {pair}: {str(e)}", exc_info=True)
            dataframe["enter_long"] = 0
            dataframe["enter_short"] = 0
            return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Определение сигналов на выход из позиции на основе анализа кластеров ликвидности
        """
        try:
            pair = metadata.get("pair", "unknown")

            # Инициализируем колонки для сигналов выхода
            dataframe["exit_long"] = 0
            dataframe["exit_short"] = 0

            # Проверяем наличие необходимых колонок
            required_columns = [
                "buy_cluster_volume",
                "sell_cluster_volume",
                "close",
                "buy_cluster_price",
                "sell_cluster_price",
            ]

            if not all(col in dataframe.columns for col in required_columns):
                logger.warning(f"Отсутствуют необходимые колонки для анализа выхода в паре {pair}")
                return dataframe

            # Получаем последнюю строку с данными
            last_row = dataframe.iloc[-1]

            # Рассчитываем дополнительные метрики для анализа
            buy_pressure = last_row["buy_cluster_volume"] / (last_row["sell_cluster_volume"] or 1)
            sell_pressure = last_row["sell_cluster_volume"] / (last_row["buy_cluster_volume"] or 1)

            # Условия для выхода из длинной позиции
            exit_long_conditions = [
                # Резкое увеличение объема продаж
                sell_pressure > 2.0,
                # Цена упала ниже цены кластера покупок с учетом спреда
                last_row["close"] < last_row["buy_cluster_price"] * 0.995,
                # Объем кластера продаж значительно вырос
                last_row["sell_cluster_volume"] > last_row["buy_cluster_volume"] * 1.8,
            ]

            # Условия для выхода из короткой позиции
            exit_short_conditions = [
                # Резкое увеличение объема покупок
                buy_pressure > 2.0,
                # Цена поднялась выше цены кластера продаж с учетом спреда
                last_row["close"] > last_row["sell_cluster_price"] * 1.005,
                # Объем кластера покупок значительно вырос
                last_row["buy_cluster_volume"] > last_row["sell_cluster_volume"] * 1.8,
            ]

            # Применяем условия к последней свече
            if all(exit_long_conditions):
                dataframe.loc[dataframe.index[-1], "exit_long"] = 1
                logger.info(
                    f"СИГНАЛ НА ВЫХОД ИЗ ПОКУПКИ {pair} | "
                    f"Цена: {last_row['close']:.8f} | "
                    f"Давление продавцов: {sell_pressure:.2f}x | "
                    f"Объем продаж: {last_row['sell_cluster_volume']:.4f} (покупки: {last_row['buy_cluster_volume']:.4f})"
                )

            if all(exit_short_conditions):
                dataframe.loc[dataframe.index[-1], "exit_short"] = 1
                logger.info(
                    f"СИГНАЛ НА ВЫХОД ИЗ ПРОДАЖИ {pair} | "
                    f"Цена: {last_row['close']:.8f} | "
                    f"Давление покупателей: {buy_pressure:.2f}x | "
                    f"Объем покупок: {last_row['buy_cluster_volume']:.4f} (продажи: {last_row['sell_cluster_volume']:.4f})"
                )

            return dataframe

        except Exception as e:
            logger.error(f"Ошибка в populate_exit_trend для {pair}: {str(e)}", exc_info=True)
            dataframe["exit_long"] = 0
            dataframe["exit_short"] = 0
            return dataframe

    def rsi(self, dataframe: DataFrame, timeperiod: int = 14) -> Series:
        """
        Расчет индекса относительной силы (RSI)
        """
        delta = dataframe["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=timeperiod).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=timeperiod).mean()

        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def custom_entry_price(
        self,
        pair: str,
        current_time: datetime,
        proposed_rate: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        """
        Настройка цены входа в позицию
        """
        # Получаем текущий стакан
        orderbook = self.dp.exchange.fetch_l2_order_book(pair, 1)

        if side == "long":
            # Для длинной позиции ставим ордер чуть выше кластера покупок
            return proposed_rate * 1.001  # На 0.1% выше
        else:
            # Для короткой позиции ставим ордер чуть ниже кластера продаж
            return proposed_rate * 0.999  # На 0.1% ниже
