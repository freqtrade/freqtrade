from pycoingecko import CoinGeckoAPI


class FtCoinGeckoApi(CoinGeckoAPI):
    """
    Simple wrapper around pycoingecko's api to support Demo API keys.

    """

    __API_URL_BASE = "https://api.coingecko.com/api/v3/"
    __PRO_API_URL_BASE = "https://pro-api.coingecko.com/api/v3/"
    _api_key: str = ""

    def __init__(self, api_key: str = "", *, is_demo=True, retries=5):
        super().__init__(retries=retries)
        # Doint' pass api_key to parent, instead set the header on the session directly
        self._api_key = api_key

        if api_key and not is_demo:
            self.api_base_url = self.__PRO_API_URL_BASE
            self.session.params.update({"x_cg_pro_api_key": api_key})
        else:
            # Use demo api key
            self.api_base_url = self.__API_URL_BASE
            if api_key:
                self.session.params.update({"x_cg_demo_api_key": api_key})
