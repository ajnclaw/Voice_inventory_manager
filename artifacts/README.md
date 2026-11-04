# artifacts/

Drop test pictures here (JPEG, PNG or WebP -- e.g. phone photos of spare
parts). The photo tests in `tests/test_item_photo_and_prices.py` upload
every picture found in this folder to a throwaway item in a temporary
database, then check the server stores it and serves the same bytes back.

Nothing in here is committed (see `.gitignore`), and the tests never write
to your real `inventory.db` or `product_images/`.
