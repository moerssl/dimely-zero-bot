javascript: (() => {
  try {
    var style = document.createElement("style");
    var interval = null;
    var results = 0;
    style.innerHTML = "#body grid ft.hidden { display: block !important; } ";
    style.innerHTML += "grid row bd:has(.ntime div:first-child:contains('3:55pm')) { display: none; }";
    style.id = "oa-bookmarklet-style";
    document.head.appendChild(style);
    function isVisible(element) {
      if (!element) return false;
      var rect = element.getBoundingClientRect();
      return (
        rect.width > 0 &&
        rect.height > 0 &&
        window.getComputedStyle(element).display !== "none"
      );
    }
    function hideRows() {
        document.querySelectorAll('.ntime div:first-child').forEach(el => {
            if (el.textContent.trim().startsWith("3:5")) {
                el.closest(' grid row bd').style.display = "none";
            }
        });
    }
    function clickVisibleButton() {
      console.log("Checking for visible button...");
      var buttons = document.querySelectorAll("a.loadmore.isready");
      var rows = document.querySelectorAll("view grid row");
      console.log("Found " + buttons.length + " buttons and " + rows.length + " rows.");


      
      if (rows.length == results) {
        console.log("No new rows found, stopping bookmarklet.");
        clearInterval(interval);
        alert("All loaded");

        return;
      }
      results = rows.length;
      for (let btn of buttons) {
        if (isVisible(btn)) {
          console.log("Clicking visible button!");
          btn.click();
          break;
        }
      }

      setTimeout(hideRows, 2000)
    }
    clickVisibleButton();
    interval = setInterval(clickVisibleButton, 2500);
    console.log("Bookmarklet activated!");
  } catch (error) {
    console.error("Error in bookmarklet:", error);
  }
})();
