$(document).ready(function () {
    $('#post-form').submit(function (event) {
        event.preventDefault()
        var input = $('#input').val();
        var csrfToken = $('input[name="csrfmiddlewaretoken"]').val();
        $.ajax({
            type: 'POST',
            url: '/predict/',
            data: {
                'input': input,
                'csrfmiddlewaretoken': csrfToken
            },
            success: function (response) {
                const body = document.querySelector('.body')
                var xValues = ["Computer Science", "Physics", "Mathematics", "Statistics", "Quantitative Biology","Quantitative Finance"];
                var yValues = response.answer
                var name_catagory_h4=document.querySelector('.name_catagory')
                var name_string="Catagory: "
                response.name_catagory.forEach(function(element) {
                    name_string+=element+", "
                  });
                
                name_catagory_h4.innerHTML=name_string
                var barColors = ["red", "green","blue","orange","brown"];

                new Chart("myChart", {
                type: "bar",
                data: {
                    labels: xValues,
                    datasets: [{
                    backgroundColor: barColors,
                    data: yValues
                    }]
                },
                options: {
                    legend: {display: false},
                    title: {
                    display: true,
                    text: "Classification Result"
                    }
                }
                });
            },
            error: function (xhr, errmsg, err) {
                console.log(xhr.status + ': ' + xhr.responseText);
            }
        })
    })
})

