curl ^"https://...^" ^
  -H ^"accept: */*^" ^
  -H ^"accept-language: en-DE,en;q=0.9,de-DE;q=0.8,de;q=0.7,en-US;q=0.6^" ^
  -H ^"content-type: application/json^" ^
  -b ^"_ga=GA1.1.978030750.1738927304; a5sid=f0ae7b80-3312-11f0-900c-3ded46d1b66c; _a5pid=b8e91b2e6c7fc2c6bdf16c71373b22b9; _a5sid=f0ae7b80-3312-11f0-900c-3ded46d1b66c; _iidt=BgevKFAUjmwmdK+Ym7hz+9qF2S3iYGt8aA8W3xPPVIa/39fnj65yVnoOfw/I4qX1ZgTwHXOVxWajEx7cnZfzpurmwopVemuflupdVHnOzc6uDilsJHlPMlzD; _vid_t=PKdVAN98mPAkUHWhiQUZPFQZq+iEiM/IpskVvAWE228CqjwH3DYMGUY6t9SO8Ed83bemzZ8eQZryxGUPigibOBIPjjv3wjvTSxvnnvyPd+jXk4bB5xGIhjEb; _a5fid=QpcFBPiSR70zHdsUnkgN; _gcl_au=1.1.1892286365.1747898891.1433286939.1748795994.1748795994; a5sid=0beae5f0-3f07-11f0-9081-794d03d657c8; _a5uid=Sba3Y2rPE; _ga_BWVV7MDW9C=GS2.1.s1749128055^$o15^$g0^$t1749128055^$j60^$l0^$h0; cf_clearance=57IKH4tcJjbIRAXOcBRu5sJoIPp5KzU4CNuyShvU5Qk-1749211434-1.2.1.1-UZpqDmjZ8MfUhWbLlaMrx8H9x7DtKZb.awU4krW7aBo8vKjDTUhDlKu7oEVEZQyUbN9NmkRqfN1yTf44jC2oOnWcCsywEi8VL75knwrwz3wK2rzW3jxrm4pPz26OU3rpRC6zmS0xwq2E9cGEdjSXTgsnIHeTdOLow8rya4nSpNqsnujI7vQzjaQOjhhwwet3hOIF79C5RKe888RR9N1_vVpHBLNXXQ0Ljhx8NvwVFoDGRbwzTMxRywLP2.mIPxs9.uF3ff7v44h1xiCd5IN_hx.QEKbYAygSKYkzjbQUZoQ5F2pz6BupDMV.6G5IYnBALv9P2hQElRd.kLPjx7OIaqe9jOfJj95UO4WZQZyh4QA^" ^
  -H ^"dnt: 1^" ^
  -H ^"origin: https://app.optionalpha.com^" ^
  -H ^"priority: u=1, i^" ^
  -H ^"referer: https://app.optionalpha.com/zdte/backtester/toptests?symbol=SPY^%^2CQQQ^%^2CXSP^%^2CIWM^&strategy=shortcallspread^%^2Cshortputspread^%^2Cironcondor^%^2Cironbutterfly^&period=3y^&winrate=65^&ror=10^&trades=300^&sort=pfactor^%^2Cdesc^" ^
  -H ^"sec-ch-ua: ^\^"Google Chrome^\^";v=^\^"137^\^", ^\^"Chromium^\^";v=^\^"137^\^", ^\^"Not/A)Brand^\^";v=^\^"24^\^"^" ^
  -H ^"sec-ch-ua-mobile: ?0^" ^
  -H ^"sec-ch-ua-platform: ^\^"Windows^\^"^" ^
  -H ^"sec-fetch-dest: empty^" ^
  -H ^"sec-fetch-mode: cors^" ^
  -H ^"sec-fetch-site: same-origin^" ^
  -H ^"user-agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36^" ^
  --data-raw ^"^[^{^\^"t^\^":^\^"rpc^\^",^\^"tid^\^":^\^"1749212494418-10135^\^",^\^"api^\^":^\^"zdte.toptests^\^",^\^"args^\^":^[^{^\^"where^\^":^{^\^"symbol^\^":^\^"SPY,QQQ,XSP,IWM^\^",^\^"strategy^\^":^\^"shortcallspread,shortputspread,ironcondor,ironbutterfly^\^",^\^"period^\^":^\^"3y^\^",^\^"trades^\^":300,^\^"winrate^\^":65,^\^"ror^\^":10^},^\^"start^\^":0,^\^"limit^\^":1,^\^"order^\^":^[^\^"pfactor^\^",^\^"desc^\^"^]^}^]^}^]^" | jq -r '
  .data as $data |
  ($data[0] | keys_unsorted) as $keys |
  ($keys | @csv), 
  ($data[] | [.[$keys[]]] | @csv)
' > ../oa.csv