#include "timeline_extension.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "absl/time/civil_time.h"
#include "civil_day_caster.h"
#include "reader.h"

namespace py = pybind11;

struct PatientDay {
    absl::CivilDay date;
    uint32_t age;
    std::vector<uint32_t> observations;
    std::vector<ObservationWithValue> observations_with_values;

    absl::Span<const uint32_t> get_observations() const { return observations; }

    absl::Span<const ObservationWithValue> get_observations_with_values()
        const {
        return observations_with_values;
    }
};

struct Patient {
    uint32_t patient_id;
    std::vector<PatientDay> days;

    absl::Span<const PatientDay> get_days() const { return days; }
};

class TimelineReader {
   public:
    class TimelineReaderIterator {
       public:
        TimelineReaderIterator(TimelineReader* r, std::vector<uint32_t> p,
                               std::optional<absl::CivilDay> s,
                               std::optional<absl::CivilDay> e)
            : reader(r),
              patient_ids(std::move(p)),
              index(0),
              start_date(s),
              end_date(e) {}

        Patient next() {
            if (index == patient_ids.size()) {
                throw py::stop_iteration();
            } else {
                return reader->get_patient(patient_ids[index++], start_date,
                                           end_date);
            }
        }

       private:
        TimelineReader* reader;
        std::vector<uint32_t> patient_ids;
        uint32_t index;
        std::optional<absl::CivilDay> start_date;
        std::optional<absl::CivilDay> end_date;
    };

    TimelineReader(const char* filename, bool readall)
        : reader(filename, readall), iter(reader.iter()) {}

    absl::Span<const uint32_t> get_patient_ids() const {
        return reader.get_patient_ids();
    }

    absl::Span<const uint64_t> get_original_patient_ids() const {
        return reader.get_original_patient_ids();
    }

    Patient get_patient(uint32_t patient_id,
                        std::optional<absl::CivilDay> start_date,
                        std::optional<absl::CivilDay> end_date) {
        Patient patient;
        patient.patient_id = patient_id;

        iter.process_patient(
            patient_id, [&start_date, &end_date, &patient](
                            absl::CivilDay birth_date, uint32_t age,
                            const std::vector<uint32_t>& observations,
                            const std::vector<ObservationWithValue>&
                                observations_with_values) {
                PatientDay day;
                day.date = birth_date + age;

                if (start_date && day.date < *start_date) {
                    return;
                }

                if (end_date && day.date >= *end_date) {
                    return;
                }

                day.age = age;
                day.observations = observations;
                day.observations_with_values = observations_with_values;
                patient.days.push_back(std::move(day));
            });

        return patient;
    }

    TimelineReaderIterator get_patients(
        std::optional<std::vector<uint32_t>> patient_ids,
        std::optional<absl::CivilDay> start_date,
        std::optional<absl::CivilDay> end_date) {
        if (!patient_ids) {
            patient_ids =
                std::vector<uint32_t>(std::begin(reader.get_patient_ids()),
                                      std::end(reader.get_patient_ids()));
        }
        return TimelineReaderIterator(this, std::move(*patient_ids), start_date,
                                      end_date);
    }

    const TermDictionary* get_dictionary() { return &reader.get_dictionary(); }

    const TermDictionary* get_value_dictionary() {
        return &reader.get_value_dictionary();
    }

   private:
    ExtractReader reader;
    ExtractReaderIterator iter;
};

template <typename T>
constexpr auto type_name() {
    std::string_view name, prefix, suffix;
#ifdef __clang__
    name = __PRETTY_FUNCTION__;
    prefix = "auto type_name() [T = ";
    suffix = "]";
#elif defined(__GNUC__)
    name = __PRETTY_FUNCTION__;
    prefix = "constexpr auto type_name() [with T = ";
    suffix = "]";
#elif defined(_MSC_VER)
    name = __FUNCSIG__;
    prefix = "auto __cdecl type_name<";
    suffix = ">(void)";
#endif
    name.remove_prefix(prefix.size());
    name.remove_suffix(suffix.size());
    return name;
}

namespace detail {
template <typename L, typename R>
struct has_operator_equals_impl {
    template <typename T = L,
              typename U = R>  // template parameters here to enable SFINAE
    static auto test(T&& t, U&& u)
        -> decltype(t == u, void(), std::true_type{});
    static auto test(...) -> std::false_type;
    using type = decltype(test(std::declval<L>(), std::declval<R>()));
};
}  // namespace detail

template <typename L, typename R = L>
struct has_operator_equals : detail::has_operator_equals_impl<L, R>::type {};

template <typename T, typename std::enable_if<has_operator_equals<
                          typename T::value_type>::value>::type* = nullptr>
void register_iterable(py::module& m) {
    py::class_<T>(m, std::string(type_name<T>()).c_str())
        .def(
            "__iter__",
            [](const T& span) {
                return py::make_iterator(std::begin(span), std::end(span));
            },
            py::keep_alive<0, 1>())
        .def("__len__", [](const T& span) { return span.size(); })
        .def("__getitem__",
             [](const T& span, ssize_t index) {
                 if (index < 0) {
                     index = span.size() + index;
                 }
                 return span[index];
             })
        .def("__contains__",
             [](const T& span, const typename T::value_type& value) {
                 return std::find(std::begin(span), std::end(span), value) !=
                        std::end(span);
             });
}

template <typename T, typename std::enable_if<!has_operator_equals<
                          typename T::value_type>::value>::type* = nullptr>
void register_iterable(py::module& m) {
    py::class_<T>(m, std::string(type_name<T>()).c_str())
        .def(
            "__iter__",
            [](const T& span) {
                return py::make_iterator(std::begin(span), std::end(span));
            },
            py::keep_alive<0, 1>())
        .def("__len__", [](const T& span) { return span.size(); })
        .def("__getitem__", [](const T& span, ssize_t index) {
            if (index < 0) {
                index = span.size() + index;
            }
            return span[index];
        });
}

void register_timeline_extension(py::module& root) {
    register_iterable<absl::Span<const uint32_t>>(root);
    register_iterable<absl::Span<const uint64_t>>(root);
    register_iterable<absl::Span<const PatientDay>>(root);
    register_iterable<absl::Span<const ObservationWithValue>>(root);

    py::module m = root.def_submodule("timeline");

    py::class_<TimelineReader>(m, "TimelineReader")
        .def(py::init<const char*, bool>(), py::arg("filename"),
             py::arg("readall") = false)
        .def("get_patient", &TimelineReader::get_patient, py::arg("patient_id"),
             py::arg("start_date") = std::nullopt,
             py::arg("end_date") = std::nullopt)
        .def("get_patients", &TimelineReader::get_patients,
             py::keep_alive<0, 1>(), py::arg("patient_ids") = std::nullopt,
             py::arg("start_date") = std::nullopt,
             py::arg("end_date") = std::nullopt)
        .def("get_patient_ids", &TimelineReader::get_patient_ids,
             py::keep_alive<0, 1>())
        .def("get_original_patient_ids",
             &TimelineReader::get_original_patient_ids, py::keep_alive<0, 1>())
        .def("get_dictionary", &TimelineReader::get_dictionary,
             py::return_value_policy::reference_internal)
        .def("get_value_dictionary", &TimelineReader::get_value_dictionary,
             py::return_value_policy::reference_internal);

    py::class_<TermDictionary>(m, "TermDictionary")
        .def("map", &TermDictionary::map, py::arg("code_index"))
        .def("get_word", &TermDictionary::get_word, py::arg("code"))
        .def("get_items", &TermDictionary::decompose);

    py::class_<TimelineReader::TimelineReaderIterator>(m,
                                                       "TimelineReaderIterator")
        .def("__iter__",
             [](TimelineReader::TimelineReaderIterator& it)
                 -> TimelineReader::TimelineReaderIterator& { return it; })
        .def("__next__", &TimelineReader::TimelineReaderIterator::next);

    py::class_<Patient>(m, "Patient")
        .def_readonly("patient_id", &Patient::patient_id)
        .def_property_readonly("days", &Patient::get_days,
                               py::keep_alive<0, 1>());

    py::class_<PatientDay>(m, "PatientDay")
        .def_readonly("date", &PatientDay::date)
        .def_readonly("age", &PatientDay::age)
        .def_property_readonly("observations", &PatientDay::get_observations,
                               py::keep_alive<0, 1>())
        .def_property_readonly("observations_with_values",
                               &PatientDay::get_observations_with_values,
                               py::keep_alive<0, 1>());

    py::class_<ObservationWithValue>(m, "ObservationWithValue")
        .def_readonly("code", &ObservationWithValue::code)
        .def_readonly("is_text", &ObservationWithValue::is_text)
        .def_readonly("text_value", &ObservationWithValue::text_value)
        .def_readonly("numeric_value", &ObservationWithValue::numeric_value);
}